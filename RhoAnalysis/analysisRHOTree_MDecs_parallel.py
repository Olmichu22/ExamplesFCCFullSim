"""
analysisRHOTree_MDecs_parallel.py

Versión reco-level del árbol MDecs. Lee ficheros EDM4hep, reconstruye ambos
hemisferios tau (hadrónicos via extractTauDecays + leptónicos via muonReco/
electronReco), los empareja con gen-taus por dR, y escribe un TTree con la
misma estructura tau1_*/tau2_* que genOnlyRHOTree_MDecs_parallel.py.

Compatible directamente con RhoHistFromTree_MDecs_parallel.py.

USO:
    python analysisRHOTree_MDecs_parallel.py \\
        -c config/default/taurecolong.yaml \\
        --decay-pair 2 -13 \\
        --n-workers 4
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

from modules.TauDecays import extractTauDecays
from modules import myutils, tauReco
from modules import optimalVariabRho, weightsPol


# ── Config ────────────────────────────────────────────────────────────────────

_DEFAULT_CONFIG = "config/default/taurecolong.yaml"
_OUTPUT_BASE    = "Results/RhoAnalysis/"

_SHARED_VARIABS = ["GenZMass", "GenZVisMass", "ZMass", "beamE"]

_TAU_SCALAR_SUFFIXES = [
    "P", "E", "M", "Theta", "Phi",
    "visP", "visE", "visM", "visTheta", "visPhi",
    "pionP", "pionE", "pionM", "pionTheta", "pionPhi",
    "lepP", "lepE", "lepTheta", "lepPhi", "lepPDG",
    "decayID", "cos_theta", "cos_psi", "cos_beta",
    "omega", "weight_P1", "weight_M1",
    "cos_theta_tau", "optimalVar", "isElectron", "nPhotons",
    "recoVisP", "recoVisE", "recoVisM", "recoVisTheta", "recoVisPhi",
    "recoPionP", "recoPionE", "recoPionM", "recoPionTheta", "recoPionPhi",
    "recoTauID", "recoLepP", "recoLepE", "recoLepTheta", "recoLepPhi", "recoLepPDG",
]

_TAU_VECTOR_SUFFIXES = [
    "photons_E", "photons_theta", "photons_phi",
    "reco_photons_E", "reco_photons_theta", "reco_photons_phi",
]

_VARIABS = (
    _SHARED_VARIABS
    + [f"tau1_{s}" for s in _TAU_SCALAR_SUFFIXES]
    + [f"tau2_{s}" for s in _TAU_SCALAR_SUFFIXES]
)
_VECTOR_VARIABS = (
    [f"tau1_{s}" for s in _TAU_VECTOR_SUFFIXES]
    + [f"tau2_{s}" for s in _TAU_VECTOR_SUFFIXES]
)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _remap_to_gen(decay_pair):
    """Convierte IDs reco a IDs gen: 2→1. Devuelve frozenset."""
    return frozenset(1 if d == 2 else d for d in decay_pair)


def split_filenames(filenames, n_workers):
    k, rem = divmod(len(filenames), n_workers)
    chunks, start = [], 0
    for i in range(n_workers):
        end = start + k + (1 if i < rem else 0)
        if start < end:
            chunks.append(filenames[start:end])
        start = end
    return chunks


def split_mlpf(mlpf_results, file_chunks):
    mlpf_chunks = []
    file_offset = 0
    for chunk in file_chunks:
        lo = file_offset * 1000
        hi = (file_offset + len(chunk)) * 1000
        sub = {k - lo: v for k, v in mlpf_results.items() if lo <= k < hi}
        mlpf_chunks.append(sub)
        file_offset += len(chunk)
    return mlpf_chunks


def _setup_worker_logging(outputpath, worker_id):
    root_logger = logging.getLogger()
    for h in root_logger.handlers[:]:
        root_logger.removeHandler(h)
        h.close()
    log_file = os.path.join(outputpath, f"worker_{worker_id}.log")
    logging.basicConfig(
        filename=log_file, level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s", force=True,
    )
    lg = logging.getLogger(f"worker_{worker_id}")
    return {"processing": lg, "io": lg, "config": lg}


def _make_p4_from_const(const):
    p4 = ROOT.TLorentzVector()
    try:
        p4.SetXYZM(const.getMomentum().x, const.getMomentum().y,
                   const.getMomentum().z, const.getMass())
    except AttributeError:
        p4.SetXYZM(const.getMomentum().X(), const.getMomentum().Y(),
                   const.getMomentum().Z(), const.getMass())
    return p4


def _extract_pion_p4(daughters):
    """Extrae el pion cargado (|PDG|==211) de un dict de daughters."""
    pion = ROOT.TLorentzVector()
    pion.SetXYZM(0, 0, 0, 0)
    if daughters is None:
        return pion
    for key in daughters:
        if abs(daughters[key].getPDG()) == 211:
            pion = _make_p4_from_const(daughters[key])
            break
    return pion


def _build_reco_candidates(recoTaus, recoElectrons, recoMuons,
                            minPTauElectron, minPTauMuon):
    """
    Combina hadrónicos, muones y electrones en una lista uniforme.
    Cada entrada: {tau_id, vis_p4, pion_p4, lep_p4, is_electron, consts}
    """
    candidates = []

    for tau in recoTaus:
        consts = tau.getDaughters()
        candidates.append({
            "tau_id":      tau.getID(),
            "vis_p4":      tau.getMomentum(),
            "pion_p4":     _extract_pion_p4(consts),
            "lep_p4":      None,
            "is_electron": False,
            "consts":      consts,
        })

    for mu_key in recoMuons:
        mu_p4 = recoMuons[mu_key].getMomentum()
        if mu_p4.P() > minPTauMuon:
            candidates.append({
                "tau_id": -13, "vis_p4": mu_p4,
                "pion_p4": None, "lep_p4": mu_p4,
                "is_electron": False, "consts": {},
            })

    for e_key in recoElectrons:
        e_p4 = recoElectrons[e_key].getMomentum()
        if e_p4.P() > minPTauElectron:
            candidates.append({
                "tau_id": -11, "vis_p4": e_p4,
                "pion_p4": None, "lep_p4": e_p4,
                "is_electron": True, "consts": {},
            })

    return candidates


def _select_decay_pair(candidates, decay_pair):
    """
    Devuelve (cand0, cand1) cuyas tau_ids coinciden con decay_pair[0] y decay_pair[1].
    Para pares simétricos requiere 2 candidatos del mismo tipo.
    Ambos seleccionados por mayor P. Devuelve (None, None) si no hay par válido.
    """
    id0, id1 = decay_pair
    pool0 = [c for c in candidates if c["tau_id"] == id0]
    pool1 = [c for c in candidates if c["tau_id"] == id1]

    if id0 == id1:
        if len(pool0) < 2:
            return None, None
        sorted_pool = sorted(pool0, key=lambda c: c["vis_p4"].P(), reverse=True)
        return sorted_pool[0], sorted_pool[1]

    if not pool0 or not pool1:
        return None, None

    cand0 = max(pool0, key=lambda c: c["vis_p4"].P())
    cand1 = max(pool1, key=lambda c: c["vis_p4"].P())
    return cand0, cand1


def _match_gen_tau(vis_p4, genTaus, used_indices):
    """Devuelve el índice gen más cercano en dR, excluyendo used_indices. -1 si vacío."""
    best_dr, best_idx = 10.0, -1
    for g, tau in genTaus.items():
        if g in used_indices:
            continue
        dr = myutils.dRAngle(tau.getMomentum(), vis_p4)
        if dr < best_dr:
            best_dr, best_idx = dr, g
    return best_idx


def _fill_tau_branches_mdecs(branches, prefix, reco_cand, gen_tau_obj, beamE, sin_eff):
    """
    Rellena las ramas {prefix}_* de un hemisferio.
    Ramas gen  → de gen_tau_obj (si no es None).
    Ramas reco → de reco_cand.
    """
    # ── Reco branches ────────────────────────────────────────────────────────
    vis_p4  = reco_cand["vis_p4"]
    pion_p4 = reco_cand["pion_p4"] if reco_cand["pion_p4"] is not None else ROOT.TLorentzVector()
    lep_p4  = reco_cand["lep_p4"]  if reco_cand["lep_p4"]  is not None else ROOT.TLorentzVector()
    reco_id = reco_cand["tau_id"]

    branches[f"{prefix}_recoTauID"].value     = float(reco_id)
    branches[f"{prefix}_recoVisP"].value      = vis_p4.P()
    branches[f"{prefix}_recoVisE"].value      = vis_p4.E()
    branches[f"{prefix}_recoVisM"].value      = vis_p4.M()
    branches[f"{prefix}_recoVisTheta"].value  = vis_p4.Theta()
    branches[f"{prefix}_recoVisPhi"].value    = vis_p4.Phi()
    branches[f"{prefix}_recoPionP"].value     = pion_p4.P()
    branches[f"{prefix}_recoPionE"].value     = pion_p4.E()
    branches[f"{prefix}_recoPionM"].value     = pion_p4.M()
    branches[f"{prefix}_recoPionTheta"].value = pion_p4.Theta()
    branches[f"{prefix}_recoPionPhi"].value   = pion_p4.Phi()

    is_lep_reco = reco_id in (-11, -13)
    branches[f"{prefix}_recoLepP"].value     = lep_p4.P()     if is_lep_reco else 0.0
    branches[f"{prefix}_recoLepE"].value     = lep_p4.E()     if is_lep_reco else 0.0
    branches[f"{prefix}_recoLepTheta"].value = lep_p4.Theta() if is_lep_reco else 0.0
    branches[f"{prefix}_recoLepPhi"].value   = lep_p4.Phi()   if is_lep_reco else 0.0
    branches[f"{prefix}_recoLepPDG"].value   = float(abs(reco_id)) if is_lep_reco else 0.0

    # Fotones reco (hadrónicos)
    for const_key, const in reco_cand["consts"].items():
        if abs(const.getPDG()) == 22:
            ph = _make_p4_from_const(const)
            branches[f"{prefix}_reco_photons_E"].push_back(ph.E())
            branches[f"{prefix}_reco_photons_theta"].push_back(ph.Theta())
            branches[f"{prefix}_reco_photons_phi"].push_back(ph.Phi())

    # ── Gen branches ─────────────────────────────────────────────────────────
    if gen_tau_obj is None:
        _no_gen_defaults = {
            "omega": -999.0, "optimalVar": -999.0,
        }
        for sfx in _TAU_SCALAR_SUFFIXES:
            if sfx.startswith("reco"):
                continue
            val = _no_gen_defaults.get(sfx, 0.0)
            branches[f"{prefix}_{sfx}"].value = val
        return

    decayID = gen_tau_obj.getID()
    tauP4   = gen_tau_obj.getMomentum()
    gen_vis = gen_tau_obj.getvisMomentum()

    branches[f"{prefix}_P"].value           = tauP4.P()
    branches[f"{prefix}_E"].value           = tauP4.E()
    branches[f"{prefix}_M"].value           = tauP4.M()
    branches[f"{prefix}_Theta"].value       = tauP4.Theta()
    branches[f"{prefix}_Phi"].value         = tauP4.Phi()
    branches[f"{prefix}_cos_theta_tau"].value = math.cos(tauP4.Theta())
    branches[f"{prefix}_visP"].value        = gen_vis.P()
    branches[f"{prefix}_visE"].value        = gen_vis.E()
    branches[f"{prefix}_visM"].value        = gen_vis.M()
    branches[f"{prefix}_visTheta"].value    = gen_vis.Theta()
    branches[f"{prefix}_visPhi"].value      = gen_vis.Phi()
    branches[f"{prefix}_decayID"].value     = float(decayID)

    # Daughters: pion cargado + fotones gen
    daughters  = gen_tau_obj.getDaughters()
    gen_pion   = ROOT.TLorentzVector()
    gen_pion.SetXYZM(0, 0, 0, 0)
    for key in daughters:
        if abs(daughters[key].getPDG()) == 211:
            gen_pion = _make_p4_from_const(daughters[key])
            break

    branches[f"{prefix}_pionP"].value     = gen_pion.P()
    branches[f"{prefix}_pionE"].value     = gen_pion.E()
    branches[f"{prefix}_pionM"].value     = gen_pion.M()
    branches[f"{prefix}_pionTheta"].value = gen_pion.Theta()
    branches[f"{prefix}_pionPhi"].value   = gen_pion.Phi()

    n_gen_photons = 0.0
    for key in daughters:
        if abs(daughters[key].getPDG()) == 111:
            for ph in daughters[key].getDaughters():
                if ph.getGeneratorStatus() != 1:
                    continue
                n_gen_photons += 1
                phP4 = _make_p4_from_const(ph)
                branches[f"{prefix}_photons_E"].push_back(phP4.E())
                branches[f"{prefix}_photons_theta"].push_back(phP4.Theta())
                branches[f"{prefix}_photons_phi"].push_back(phP4.Phi())
    branches[f"{prefix}_nPhotons"].value = n_gen_photons

    # Polarización (dispatch por decayID)
    if decayID == 1:
        (cos_theta, cos_psi, cos_beta, omega,
         w_P1, w_M1) = optimalVariabRho.wVariab(
            tauP4, gen_vis, gen_pion, beamE, sin_eff=sin_eff)
        opt_var = omega
    elif decayID in (0, 10):
        w_P1 = weightsPol.newAtau(tauP4, gen_vis, decayID, +1, sin_eff=sin_eff)
        w_M1 = weightsPol.newAtau(tauP4, gen_vis, decayID, -1, sin_eff=sin_eff)
        cos_theta = cos_psi = cos_beta = 0.0
        omega = -999.0
        opt_var = gen_pion.E() / tauP4.E() if tauP4.E() > 0 else -999.0
    elif decayID in (-11, -13):
        w_P1 = weightsPol.newAtauLep(gen_vis, tauP4, beamE, +1, sin_eff=sin_eff)
        w_M1 = weightsPol.newAtauLep(gen_vis, tauP4, beamE, -1, sin_eff=sin_eff)
        cos_theta = cos_psi = cos_beta = 0.0
        omega = -999.0
        opt_var = gen_vis.E() / tauP4.E() if tauP4.E() > 0 else -999.0
    else:
        cos_theta = cos_psi = cos_beta = omega = 0.0
        w_P1 = w_M1 = 1.0
        opt_var = -999.0

    branches[f"{prefix}_cos_theta"].value  = cos_theta
    branches[f"{prefix}_cos_psi"].value    = cos_psi
    branches[f"{prefix}_cos_beta"].value   = cos_beta
    branches[f"{prefix}_omega"].value      = omega
    branches[f"{prefix}_weight_P1"].value  = w_P1
    branches[f"{prefix}_weight_M1"].value  = w_M1
    branches[f"{prefix}_optimalVar"].value = opt_var

    # Leptón gen
    is_lep_gen = decayID in (-11, -13)
    branches[f"{prefix}_lepP"].value      = gen_vis.P()     if is_lep_gen else 0.0
    branches[f"{prefix}_lepE"].value      = gen_vis.E()     if is_lep_gen else 0.0
    branches[f"{prefix}_lepTheta"].value  = gen_vis.Theta() if is_lep_gen else 0.0
    branches[f"{prefix}_lepPhi"].value    = gen_vis.Phi()   if is_lep_gen else 0.0
    branches[f"{prefix}_lepPDG"].value    = float(abs(decayID)) if is_lep_gen else 0.0
    branches[f"{prefix}_isElectron"].value = float(decayID == -11)


# ── Worker ────────────────────────────────────────────────────────────────────

def process_chunk_mdecs(filenames_chunk, mlpf_chunk, global_event_offset,
                         config_bundle, worker_id):
    outputpath       = config_bundle["outputpath"]
    decay_pair       = config_bundle["decay_pair"]
    dRMax            = config_bundle["dRMax"]
    tauPCut          = config_bundle["tauPCut"]
    minPTauPhoton    = config_bundle["minPTauPhoton"]
    minPTauPion      = config_bundle["minPTauPion"]
    PNeutron         = config_bundle["PNeutron"]
    generalPCut      = config_bundle["generalPCut"]
    photon_config    = config_bundle["photon_config"]
    sin_eff          = config_bundle["sin_eff"]
    minPTauElectron  = config_bundle["minPTauElectron"]
    minPTauMuon      = config_bundle["minPTauMuon"]
    gen_taus_sample  = config_bundle.get("gen_taus_sample", True)
    gatr_results_path = config_bundle["gatr_results_path"]
    test_pfo         = config_bundle["test_pfo"]
    fileOutName_base  = config_bundle["fileOutName_base"]

    loggers        = _setup_worker_logging(outputpath, worker_id)
    logger_process = loggers["processing"]
    logger_io      = loggers["io"]

    # Salida parcial
    partial_path = os.path.join(outputpath,
                                f"partial_worker{worker_id}_{fileOutName_base}.root")
    partial_outfile = TFile(partial_path, "RECREATE")
    tree = TTree("outtree_original", "MDecs reco+gen variables")

    branches = {}
    for var in _VARIABS:
        branches[var] = ctypes.c_double(0.0)
        tree.Branch(var, ctypes.addressof(branches[var]), f"{var}/D")
    for var in _VECTOR_VARIABS:
        branches[var] = ROOT.std.vector("double")()
        tree.Branch(var, branches[var])

    totalEvents    = 0
    selectedEvents = 0
    eventid        = global_event_offset
    skipped_files  = []

    for filename in filenames_chunk:
        try:
            file_reader = root_io.Reader([filename])
        except Exception as exc:
            logger_io.warning("Worker %d: skipping %s: %s", worker_id, filename, exc)
            skipped_files.append(filename)
            continue

        for event in file_reader.get("events"):
            if totalEvents % 500 == 0:
                logger_process.info("Worker %d: processed %d events", worker_id, totalEvents)
            totalEvents += 1

            # Clear vectors
            for var in _VECTOR_VARIABS:
                branches[var].clear()

            mc_particles = event.get("MCParticles")
            beamE = mc_particles[0].getEnergy()

            # Gen taus
            genTaus = tauReco.findAllGenTaus(mc_particles) if gen_taus_sample else {}
            nGenTaus = len(genTaus)

            GenZMass = GenZVisMass = 0.0
            if nGenTaus == 2:
                gen_list = list(genTaus.values())
                ZGenP4  = gen_list[0].getMomentum() + gen_list[1].getMomentum()
                ZVisP4  = gen_list[0].getvisMomentum() + gen_list[1].getvisMomentum()
                GenZMass    = ZGenP4.M()
                GenZVisMass = ZVisP4.M()

            # Reco reconstruction
            pfos = event.get("PandoraPFOs")
            recoTau_raw, recoElectrons, recoMuons, _ = extractTauDecays(
                gatr_results_path, mlpf_chunk, eventid, pfos,
                dRMax, minPTauPhoton, minPTauPion, PNeutron,
                generalPCut, photon_config, False, test_pfo, logger_process,
            )
            recoTaus = myutils.sort_by_P(recoTau_raw)

            # Candidate list + pair selection
            candidates = _build_reco_candidates(
                recoTaus, recoElectrons, recoMuons, minPTauElectron, minPTauMuon)
            cand0, cand1 = _select_decay_pair(candidates, decay_pair)
            if cand0 is None:
                eventid += 1
                continue

            # Optional momentum cut
            if (cand0["vis_p4"].P() < tauPCut and cand1["vis_p4"].P() < tauPCut):
                eventid += 1
                continue

            # ZMass reco
            ZMass = (cand0["vis_p4"] + cand1["vis_p4"]).M()

            # Order tau1/tau2 by energy (tau1 = higher E)
            if cand0["vis_p4"].E() >= cand1["vis_p4"].E():
                tau1_cand, tau2_cand = cand0, cand1
            else:
                tau1_cand, tau2_cand = cand1, cand0

            # Gen matching (greedy dR, no reuse)
            used_gen = set()
            gen_idx_1 = _match_gen_tau(tau1_cand["vis_p4"], genTaus, used_gen)
            if gen_idx_1 != -1:
                used_gen.add(gen_idx_1)
            gen_idx_2 = _match_gen_tau(tau2_cand["vis_p4"], genTaus, used_gen)

            gen_tau1 = genTaus.get(gen_idx_1)
            gen_tau2 = genTaus.get(gen_idx_2)

            # Fill shared branches
            branches["GenZMass"].value    = GenZMass
            branches["GenZVisMass"].value = GenZVisMass
            branches["ZMass"].value       = ZMass
            branches["beamE"].value       = beamE

            # Fill per-tau branches
            _fill_tau_branches_mdecs(branches, "tau1", tau1_cand, gen_tau1, beamE, sin_eff)
            _fill_tau_branches_mdecs(branches, "tau2", tau2_cand, gen_tau2, beamE, sin_eff)

            selectedEvents += 1
            tree.Fill()
            eventid += 1

    if skipped_files:
        logger_io.warning("Worker %d: %d file(s) skipped", worker_id, len(skipped_files))

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
    print(f"[WARN] hadd failed (rc={result.returncode}), using TFileMerger:\n{result.stderr}")
    merger = ROOT.TFileMerger(False)
    merger.OutputFile(final_output, "RECREATE")
    for f in partial_files:
        merger.AddFile(f)
    merger.Merge()


# ── Main ──────────────────────────────────────────────────────────────────────

def my_hook(parser):
    parser.add_argument("--sin-eff", type=float, default=None,
                        help="Effective sin^2 theta_W for polarization weights")
    parser.add_argument("--n-workers", type=int, default=None,
                        help="Parallel workers (default: min(n_files, n_cpus))")
    parser.add_argument("--decay-pair", type=int, nargs=2, default=None,
                        metavar=("ID0", "ID1"),
                        help="Reco decay IDs to select (e.g. 2 -13). "
                             "Overrides general.decay_pair in YAML.")
    parser.add_argument("-e", "--electron-cut", type=float, default=10.0,
                        help="Minimum electron P [GeV] (default: 10)")
    parser.add_argument("-u", "--muon-cut", type=float, default=10.0,
                        help="Minimum muon P [GeV] (default: 10)")
    parser.add_argument("--sys-err", type=str,
                        default="config/systematics/err_sys.yml")
    parser.add_argument("--test-extremes", action="store_true",
                        help="Test photon energy resolution extremes (unused in MDecs)")


def main():
    general_configs = myutils.setup_analysis_config(_DEFAULT_CONFIG, _OUTPUT_BASE,
                                                    parser_hook=my_hook)
    loggers    = general_configs["loggers"]
    run_config = general_configs["config"]
    args       = general_configs["args"]

    # decay_pair: CLI > YAML
    decay_pair = (list(args.decay_pair) if args.decay_pair
                  else run_config.get("general", {}).get("decay_pair"))
    if not decay_pair or len(decay_pair) != 2:
        loggers["config"].error(
            "decay_pair not specified. Use --decay-pair ID0 ID1 or set "
            "general.decay_pair in the YAML config.")
        sys.exit(1)

    loggers["config"].info("decay_pair (reco): %s", decay_pair)

    dRMax         = run_config["cuts"]["dRMax"]
    tauPCut       = run_config["cuts"]["tauCut"]
    minPTauPhoton = run_config["cuts"]["TauPhotonPCut"]
    minPTauPion   = run_config["cuts"]["TauPionPCut"]
    PNeutron      = run_config["cuts"]["NeutronCut"]
    generalPCut   = run_config["cuts"]["generalPCut"]
    outputpath    = general_configs["outputpath"]

    sys_errors    = run_config.get("systematics_errors", {})
    photon_config = sys_errors.get("photon_config", {})
    sample        = run_config["general"]["sample"]
    test_arg      = general_configs["flags"]["test"]
    gen_taus_sample = general_configs.get("has_gen_taus", True)

    gatr_results_path = args.gatr_result

    decay_tag        = f"{decay_pair[0]}_{decay_pair[1]}"
    raw_fileOutName  = os.path.join(outputpath, general_configs["fileOutName"])
    fileOutName_base = f"TTree_MDecs_{decay_tag}_{Path(raw_fileOutName).stem}"
    fileOutName      = os.path.join(outputpath, f"{fileOutName_base}.root")

    filenames, mlpf_results = myutils.get_root_trees_path(
        sample, gatr_results_path, loggers, test_arg, args,
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
        "decay_pair":       decay_pair,
        "dRMax":            dRMax,
        "tauPCut":          tauPCut,
        "minPTauPhoton":    minPTauPhoton,
        "minPTauPion":      minPTauPion,
        "PNeutron":         PNeutron,
        "generalPCut":      generalPCut,
        "photon_config":    photon_config,
        "sin_eff":          args.sin_eff,
        "minPTauElectron":  args.electron_cut,
        "minPTauMuon":      args.muon_cut,
        "gen_taus_sample":  gen_taus_sample,
        "gatr_results_path": gatr_results_path,
        "test_pfo":         args.test_pfo,
        "outputpath":       outputpath,
        "fileOutName_base": fileOutName_base,
    }

    totals = {"events": 0, "selected": 0}

    if n_workers == 1:
        loggers["io"].info("Sequential mode.")
        mlpf_empty = {}
        partial_path, tot, sel = process_chunk_mdecs(
            filenames, mlpf_empty, 0, config_bundle, 0)
        import shutil
        shutil.move(partial_path, fileOutName)
        totals["events"]   = tot
        totals["selected"] = sel
        loggers["io"].info("Sequential done: events=%d selected=%d", tot, sel)
    else:
        file_chunks   = split_filenames(filenames, n_workers)
        mlpf_chunks   = split_mlpf(mlpf_results, file_chunks)
        n_chunks      = len(file_chunks)
        event_offsets = [sum(len(file_chunks[j]) for j in range(i)) * 1000
                         for i in range(n_chunks)]

        ctx           = multiprocessing.get_context("fork")
        partial_files = []
        t_start       = time.time()

        loggers["io"].info("Launching %d workers...", n_chunks)
        with ProcessPoolExecutor(max_workers=n_chunks, mp_context=ctx) as executor:
            futures = {
                executor.submit(process_chunk_mdecs,
                                file_chunks[i], mlpf_chunks[i],
                                event_offsets[i], config_bundle, i): i
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
                    print(f"  [{n_done}/{n_chunks}] worker {wid} done "
                          f"({tot} events, {elapsed:.1f}s)", flush=True)
                except Exception as exc:
                    loggers["io"].error("Worker %d failed: %s", wid, exc)
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

    # Resumen CSV + config
    results_df = pd.DataFrame({
        "TotalEvents":    [totals["events"]],
        "SelectedEvents": [totals["selected"]],
    })
    results_df.to_csv(
        os.path.join(outputpath, f"results_summary_{decay_tag}.csv"), index=False)

    output_config_file = os.path.join(outputpath, "config.yaml")
    run_config["args"]       = vars(args)
    run_config["run_Type"]   = "analysisRHOTree_MDecs"
    run_config["general"]["decay_pair"] = decay_pair
    with open(output_config_file, "w") as f:
        yaml.dump(run_config, f)

    loggers["io"].info("Config saved to %s", output_config_file)
    loggers["io"].info("Output file: %s", fileOutName)


if __name__ == "__main__":
    main()
