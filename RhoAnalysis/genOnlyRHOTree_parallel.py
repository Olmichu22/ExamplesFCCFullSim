"""
genOnlyRHOTree_parallel.py

Versión gen-only de analysisRHOTree_parallel.py.
No reconstruye taus a nivel reco. Selecciona a nivel generador eventos con
topología: un tau con el decay especificado (hadrónico) + un tau leptónico.
Las ramas reco se rellenan con los valores gen para compatibilidad directa con
rhoHistFromTree_parallel.py (los cortes meson_cut, lepton_cut, zmass_cut
operan entonces sobre cantidades gen).

USO:
    python genOnlyRHOTree_parallel.py [mismos args que analysisRHOTree_parallel.py] --n-workers N

NOTA: decay=-777 acepta cualquier decay hadrónico (τ→ρ, τ→π, τ→a1, …)
      emparejado con un tau leptónico.
"""

import ctypes
import logging
import math
import multiprocessing
import os
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
import ROOT
import yaml
from podio import root_io
from ROOT import TFile, TTree

from modules import myutils, tauReco
from modules import optimalVariabRho
from modules import weightsPol


# ── Config ────────────────────────────────────────────────────────────────────

_DEFAULT_CONFIG = "config/default/taurecolong.yaml"
_OUTPUT_BASE    = "Results/RhoAnalysis/"

# Idéntico al de analysisRHOTree_parallel — rhoHistFromTree depende de este esquema
_VARIABS = [
    "genTauP", "genMesonP", "genPionP", "genTauE",
    "genTauM", "genMesonE", "genMesonM", "genPionE",
    "genPionM", "genTauTheta", "genTauPhi", "genMesonTheta",
    "genMesonPhi", "genPionTheta", "genPionPhi", "gen_cos_theta",
    "gen_cos_psi", "gen_cos_beta", "gen_w", "weight_P1",
    "weight_M1", "genOmega", "gen_cos_theta_tau", "recoMesonP",
    "recoPionP", "recoMesonTheta", "recoMesonPhi", "recoPionTheta",
    "recoPionPhi", "recoMesonE", "recoMesonM", "recoPionE",
    "recoPionM", "cos_theta", "cos_psi", "cos_beta",
    "omega", "cos_theta_rho", "genTauID", "recoTauID",
    "ZMass", "GenZMass", "GenZVisMass", "beamE",
    "nPhotonsReco", "nPhotonsGen", "isElectron", "lepP",
    "lepE", "lepTheta", "lepPhi", "lepPDG",
    "genLepP", "genLepE", "genLepTheta", "genLepPhi", "genLepPDG",
    "genLepTauP", "genLepTauE", "genLepTauTheta", "genLepTauPhi", "genLepTauM",
    "weight_lep_P1", "weight_lep_M1", "genOptimalvarPi", "genOptimalvarLep",
    "recoOptimalvarPi", "recoOptimalvarLep",
]

_VECTOR_VARIABS = [
    "reco_photons_E", "reco_photons_theta", "reco_photons_phi",
    "gen_photons_E",  "gen_photons_theta",  "gen_photons_phi",
]

# IDs de decays leptónicos en la convención de TauDecays
_LEPTONIC_IDS = {-11, -13}


# ── Helpers ───────────────────────────────────────────────────────────────────

def split_filenames(filenames, n_workers):
    k, rem = divmod(len(filenames), n_workers)
    chunks, start = [], 0
    for i in range(n_workers):
        end = start + k + (1 if i < rem else 0)
        if start < end:
            chunks.append(filenames[start:end])
        start = end
    return chunks


def _setup_worker_logging(outputpath, worker_id):
    root_logger = logging.getLogger()
    for h in root_logger.handlers[:]:
        root_logger.removeHandler(h)
        h.close()
    log_file = os.path.join(outputpath, f"worker_{worker_id}.log")
    logging.basicConfig(
        filename=log_file,
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        force=True,
    )
    lg = logging.getLogger(f"worker_{worker_id}")
    return {"processing": lg, "io": lg, "config": lg}


def _make_p4_from_const(const):
    """Construye un TLorentzVector desde una partícula EDM4hep."""
    p4 = ROOT.TLorentzVector()
    try:
        p4.SetXYZM(const.getMomentum().x, const.getMomentum().y,
                   const.getMomentum().z, const.getMass())
    except AttributeError:
        p4.SetXYZM(const.getMomentum().X(), const.getMomentum().Y(),
                   const.getMomentum().Z(), const.getMass())
    return p4


# ── Worker ────────────────────────────────────────────────────────────────────

def process_chunk_gen_only(filenames_chunk, config_bundle, worker_id):
    """
    Procesa un chunk de ficheros ROOT a nivel gen.
    Selecciona pares (tau hadrónico con decay buscado, tau leptónico).
    Las ramas reco se rellenan con los valores gen.
    """
    outputpath       = config_bundle["outputpath"]
    selectDecay      = config_bundle["selectDecay"]
    sin_eff          = config_bundle["sin_eff"]
    fileOutName_base = config_bundle["fileOutName_base"]

    loggers        = _setup_worker_logging(outputpath, worker_id)
    logger_process = loggers["processing"]
    logger_io      = loggers["io"]

    # decay=2 es un subtipo reco del rho; a nivel gen corresponde a genTauID=1
    hadronic_filter = 1 if selectDecay == 2 else selectDecay

    genparts = "MCParticles"

    # ── Crear TTree de salida parcial ─────────────────────────────────────────
    partial_path = os.path.join(outputpath,
                                f"partial_worker{worker_id}_{fileOutName_base}.root")
    partial_outfile = TFile(partial_path, "RECREATE")

    tree = TTree("outtree_original", "gen-only processed variables")
    branches = {}
    for var in _VARIABS:
        branches[var] = ctypes.c_double(0.0)
        tree.Branch(var, ctypes.addressof(branches[var]), f"{var}/D")
    for var in _VECTOR_VARIABS:
        branches[var] = ROOT.std.vector("double")()
        tree.Branch(var, branches[var])

    totalEvents    = 0
    selectedEvents = 0
    skipped_files  = []

    # ── Event loop ────────────────────────────────────────────────────────────
    for filename in filenames_chunk:
        try:
            file_reader = root_io.Reader([filename])
        except Exception as exc:
            logger_io.warning("Worker %d: skipping unreadable file %s: %s",
                              worker_id, filename, exc)
            skipped_files.append(filename)
            continue

        for event in file_reader.get("events"):
            if totalEvents % 500 == 0:
                logger_process.info("Worker %d: processed %d events",
                                    worker_id, totalEvents)
            totalEvents += 1

            for var in _VECTOR_VARIABS:
                branches[var].clear()

            mc_particles = event.get(genparts)
            beamE = mc_particles[0].getEnergy()

            genTaus = tauReco.findAllGenTaus(mc_particles)
            if len(genTaus) < 2:
                continue

            # ── Buscar par hadrónico + leptónico ──────────────────────────────
            for g_had, genTau in genTaus.items():
                genTauID = genTau.getID()

                if genTauID in _LEPTONIC_IDS:
                    continue
                if hadronic_filter != -777 and genTauID != hadronic_filter:
                    continue

                lepTau = None
                for g_lep, candidate in genTaus.items():
                    if g_lep == g_had:
                        continue
                    if candidate.getID() in _LEPTONIC_IDS:
                        lepTau = candidate
                        break

                if lepTau is None:
                    continue

                # ── Cinemática del tau hadrónico ──────────────────────────────
                genTauP4   = genTau.getMomentum()
                genMesonP4 = genTau.getvisMomentum()
                genTauConst = genTau.getDaughters()

                genPionP4 = ROOT.TLorentzVector()
                genPionP4.SetXYZM(0, 0, 0, 0)
                for const in genTauConst:
                    if abs(genTauConst[const].getPDG()) == 211:
                        genPionP4 = _make_p4_from_const(genTauConst[const])
                        break

                # ── Variables angulares gen ───────────────────────────────────
                (gen_cos_theta, gen_cos_psi, gen_cos_beta, gen_w,
                 weight_P1, weight_M1) = optimalVariabRho.wVariab(
                    genTauP4, genMesonP4, genPionP4, beamE, sin_eff=sin_eff)
                gen_cos_theta_tau = math.cos(genTauP4.Theta())

                # ── Cinemática del tau leptónico ──────────────────────────────
                genLepTauP4 = lepTau.getMomentum()
                genLepP4     = lepTau.getvisMomentum()
                genLepPDG    = abs(lepTau.getID())
                is_electron  = (lepTau.getID() == -11)

                # ── Masa Z gen ────────────────────────────────────────────────
                ZGenP4      = genTauP4 + genLepTauP4
                ZVisP4      = genMesonP4 + genLepP4
                GenZMass    = ZGenP4.M()
                GenZVisMass = ZVisP4.M()

                # ── Fotones gen desde π⁰ ──────────────────────────────────────
                nPhotonsGen = 0.0
                for const in genTauConst:
                    if abs(genTauConst[const].getPDG()) == 111:
                        pi0const = genTauConst[const].getDaughters()
                        for photon_const in pi0const:
                            if photon_const.getGeneratorStatus() != 1:
                                continue
                            nPhotonsGen += 1
                            ph = _make_p4_from_const(photon_const)
                            branches["gen_photons_E"].push_back(ph.E())
                            branches["gen_photons_theta"].push_back(ph.Theta())
                            branches["gen_photons_phi"].push_back(ph.Phi())

                selectedEvents += 1

                # ── Ramas gen ─────────────────────────────────────────────────
                branches["genTauP"].value          = genTauP4.P()
                branches["genTauE"].value          = genTauP4.E()
                branches["genTauM"].value          = genTauP4.M()
                branches["genTauTheta"].value      = genTauP4.Theta()
                branches["genTauPhi"].value        = genTauP4.Phi()
                branches["genMesonP"].value        = genMesonP4.P()
                branches["genMesonE"].value        = genMesonP4.E()
                branches["genMesonM"].value        = genMesonP4.M()
                branches["genMesonTheta"].value    = genMesonP4.Theta()
                branches["genMesonPhi"].value      = genMesonP4.Phi()
                branches["genPionP"].value         = genPionP4.P()
                branches["genPionE"].value         = genPionP4.E()
                branches["genPionM"].value         = genPionP4.M()
                branches["genPionTheta"].value     = genPionP4.Theta()
                branches["genPionPhi"].value       = genPionP4.Phi()
                branches["gen_cos_theta"].value    = gen_cos_theta
                branches["gen_cos_psi"].value      = gen_cos_psi
                branches["gen_cos_beta"].value     = gen_cos_beta
                branches["gen_w"].value            = gen_w
                branches["genOmega"].value         = gen_w
                branches["weight_P1"].value        = weight_P1
                branches["weight_M1"].value        = weight_M1
                branches["gen_cos_theta_tau"].value = gen_cos_theta_tau
                branches["genTauID"].value         = float(genTauID)
                branches["GenZMass"].value         = GenZMass
                branches["GenZVisMass"].value      = GenZVisMass
                branches["genLepP"].value          = genLepP4.P()
                branches["genLepE"].value          = genLepP4.E()
                branches["genLepTheta"].value      = genLepP4.Theta()
                branches["genLepPhi"].value        = genLepP4.Phi()
                branches["genLepPDG"].value        = float(genLepPDG)
                branches["genLepTauP"].value       = genLepTauP4.P()
                branches["genLepTauE"].value       = genLepTauP4.E()
                branches["genLepTauTheta"].value   = genLepTauP4.Theta()
                branches["genLepTauPhi"].value     = genLepTauP4.Phi()
                branches["genLepTauM"].value       = genLepTauP4.M()
                if genLepPDG in [11, 13] and genLepTauP4.E() > 0:
                    weight_lep_P1 = weightsPol.newAtauLep(genLepP4, genLepTauP4, beamE, +1, sin_eff=sin_eff)
                    weight_lep_M1 = weightsPol.newAtauLep(genLepP4, genLepTauP4, beamE, -1, sin_eff=sin_eff)
                else:
                    weight_lep_P1 = 1.0
                    weight_lep_M1 = 1.0
                branches["weight_lep_P1"].value    = weight_lep_P1
                branches["weight_lep_M1"].value    = weight_lep_M1
                branches["nPhotonsGen"].value      = nPhotonsGen
                branches["beamE"].value            = beamE
                branches["isElectron"].value       = float(is_electron)

                # ── Ramas reco = valores gen ──────────────────────────────────
                # Permite que los cortes de rhoHistFromTree actúen sobre gen
                branches["recoMesonP"].value       = genMesonP4.P()
                branches["recoMesonE"].value       = genMesonP4.E()
                branches["recoMesonM"].value       = genMesonP4.M()
                branches["recoMesonTheta"].value   = genMesonP4.Theta()
                branches["recoMesonPhi"].value     = genMesonP4.Phi()
                branches["recoPionP"].value        = genPionP4.P()
                branches["recoPionE"].value        = genPionP4.E()
                branches["recoPionM"].value        = genPionP4.M()
                branches["recoPionTheta"].value    = genPionP4.Theta()
                branches["recoPionPhi"].value      = genPionP4.Phi()
                branches["cos_theta"].value        = gen_cos_theta
                branches["cos_psi"].value          = gen_cos_psi
                branches["cos_beta"].value         = gen_cos_beta
                branches["omega"].value            = gen_w
                branches["cos_theta_rho"].value    = math.cos(genMesonP4.Theta())
                branches["recoTauID"].value        = float(genTauID)
                branches["ZMass"].value            = GenZVisMass
                branches["lepP"].value             = genLepP4.P()
                branches["lepE"].value             = genLepP4.E()
                branches["lepTheta"].value         = genLepP4.Theta()
                branches["lepPhi"].value           = genLepP4.Phi()
                branches["lepPDG"].value           = float(genLepPDG)
                branches["nPhotonsReco"].value     = 0.0
                branches["genOptimalvarPi"].value     = genPionP4.E()/genTauP4.E() if genTauP4.E() > 0 else 0.0
                branches["genOptimalvarLep"].value    = genLepP4.E()/genLepTauP4.E() if genLepTauP4.E() > 0 else 0.0
                branches["recoOptimalvarPi"].value     = genPionP4.E()/genTauP4.E() if genTauP4.E() > 0 else 0.0
                branches["recoOptimalvarLep"].value    = genLepP4.E()/genLepTauP4.E() if genLepTauP4.E() > 0 else 0.0
                # reco_photons_* permanecen vacíos (cleared al inicio del evento)

                tree.Fill()

    if skipped_files:
        logger_io.warning("Worker %d: %d file(s) skipped due to read errors",
                          worker_id, len(skipped_files))

    partial_outfile.cd()
    tree.Write()
    partial_outfile.Close()

    logger_io.info("Worker %d: done. Events=%d Selected=%d",
                   worker_id, totalEvents, selectedEvents)
    return partial_path, totalEvents, selectedEvents


# ── Merge ─────────────────────────────────────────────────────────────────────

def merge_partial_root_files(partial_files, final_output):
    cmd = ["hadd", "-f", final_output] + partial_files
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode == 0:
        return
    print(f"[WARN] hadd falló (rc={result.returncode}), usando TFileMerger:\n{result.stderr}")
    merger = ROOT.TFileMerger(False)
    merger.OutputFile(final_output, "RECREATE")
    for f in partial_files:
        merger.AddFile(f)
    merger.Merge()


# ── Main ──────────────────────────────────────────────────────────────────────

def my_hook(parser):
    parser.add_argument("--sin-eff", type=float, default=None,
                        help="Effective sin^2 theta_W for weight calculation")
    parser.add_argument("--n-workers", type=int, default=None,
                        help="Número de procesos paralelos (default: min(n_files, n_cpus))")


def main():
    general_configs = myutils.setup_analysis_config(_DEFAULT_CONFIG, _OUTPUT_BASE,
                                                    parser_hook=my_hook)
    loggers    = general_configs["loggers"]
    run_config = general_configs["config"]
    args       = general_configs["args"]

    selectDecay = general_configs["decay"]
    outputpath  = general_configs["outputpath"]
    fileOutName = os.path.join(outputpath, general_configs["fileOutName"])
    sample      = run_config["general"]["sample"]
    test_arg    = general_configs["flags"]["test"]

    fileOutName_base = Path(fileOutName).stem

    filenames, _ = myutils.get_root_trees_path(
        sample, None, loggers, test_arg, args,
        skip_root_validation=True,
    )
    loggers["io"].info("Found %d input files", len(filenames))
    if not filenames:
        loggers["io"].error("No input files found. Aborting.")
        sys.exit(1)

    if args.input_list and len(args.input_list) == 1:
        n_workers = 1
    else:
        n_workers = args.n_workers or min(len(filenames), os.cpu_count() or 1)
    loggers["io"].info("Using %d workers for %d files", n_workers, len(filenames))

    config_bundle = {
        "selectDecay":      selectDecay,
        "sin_eff":          args.sin_eff,
        "outputpath":       outputpath,
        "fileOutName_base": fileOutName_base,
    }

    if n_workers == 1:
        loggers["io"].info("Sequential mode (n_workers=1).")
        partial_path, tot, sel = process_chunk_gen_only(filenames, config_bundle, 0)
        import shutil
        shutil.move(partial_path, fileOutName)
        loggers["io"].info("Sequential done: events=%d selected=%d", tot, sel)
        totals = {"events": tot, "selected": sel}
    else:
        file_chunks = split_filenames(filenames, n_workers)
        n_chunks    = len(file_chunks)

        ctx = multiprocessing.get_context("fork")
        partial_files = []
        totals = {"events": 0, "selected": 0}
        t_start = time.time()

        loggers["io"].info("Launching %d workers...", n_chunks)
        with ProcessPoolExecutor(max_workers=n_chunks, mp_context=ctx) as executor:
            futures = {
                executor.submit(process_chunk_gen_only,
                                file_chunks[i], config_bundle, i): i
                for i in range(n_chunks)
            }
            for n_done, future in enumerate(as_completed(futures), start=1):
                wid = futures[future]
                try:
                    partial_path, tot, sel = future.result()
                    partial_files.append(partial_path)
                    totals["events"]   += tot
                    totals["selected"] += sel
                    elapsed = time.time() - t_start
                    print(f"  [{n_done}/{n_chunks}] worker {wid} terminado "
                          f"({tot} eventos, {elapsed:.1f}s acumulado)", flush=True)
                except Exception as exc:
                    loggers["io"].error("Worker %d falló: %s", wid, exc)
                    raise

        loggers["io"].info("Merging %d partial files → %s",
                           len(partial_files), fileOutName)
        merge_partial_root_files(partial_files, fileOutName)

        for p in partial_files:
            try:
                os.remove(p)
            except OSError:
                pass

        loggers["io"].info("Total events=%d selected=%d (%.1fs)",
                           totals["events"], totals["selected"],
                           time.time() - t_start)

    decay_str = general_configs["decay"]
    results_df = pd.DataFrame({
        "TotalEvents":    [totals["events"]],
        "SelectedEvents": [totals["selected"]],
    })
    results_df.to_csv(
        os.path.join(outputpath, f"results_summary_{decay_str}.csv"), index=False)

    output_config_file = os.path.join(outputpath, "config.yaml")
    run_config["args"] = vars(args)
    run_config["run_Type"]="genOnlyRHOTree"
    with open(output_config_file, "w") as f:
        yaml.dump(run_config, f)
    loggers["io"].info("Config guardado en %s", output_config_file)
    loggers["io"].info("Fichero de salida: %s", fileOutName)


if __name__ == "__main__":
    main()
