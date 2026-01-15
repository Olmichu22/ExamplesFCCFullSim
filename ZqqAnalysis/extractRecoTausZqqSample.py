import argparse
import copy
import logging
import math
import os
import pprint

import numpy as np
import pandas as pd
import ROOT
import yaml
from podio import root_io

from modules import myutils
from modules.TauDecays import extractTauDecays

# ----------------------------------------------------------------------------
def write_histograms_recursive(obj):
    """Recorre un diccionario anidado y ejecuta `.Write()` en cada objeto tipo ROOT histogram."""
    if isinstance(obj, dict):
        for value in obj.values():
            write_histograms_recursive(value)
    else:
        try:
            obj.Write()
        except AttributeError:
            print(f"Objeto {obj} no tiene método .Write(). Ignorado.")


def safe_get(dct, *keys):
    """Accede a diccionarios anidados sin KeyError. Devuelve None si no existe."""
    cur = dct
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return None
        cur = cur[k]
    return cur


def safe_fill(root_histograms, path_keys, *fill_args):
    """
    Rellena un histograma si existe en root_histograms.
    path_keys: ("Reco","Events","NTaus") etc.
    """
    h = safe_get(root_histograms, *path_keys)
    if h is None:
        return
    try:
        h.Fill(*fill_args)
    except Exception:
        # Si por lo que sea el hist no es TH1/TH2/TGraph o cambia firma, no rompemos el job.
        return


def p4_from_particle(p):
    """
    Devuelve ROOT.TLorentzVector a partir de un objeto EDM4hep-like.
    Se intenta cubrir los dos estilos (x,y,z) y (X(),Y(),Z()) que aparecen en tu base.
    """
    p4 = ROOT.TLorentzVector()
    mom = p.getMomentum()
    mass = 0.0
    try:
        mass = float(p.getMass())
    except Exception:
        mass = 0.0

    try:
        p4.SetXYZM(float(mom.x), float(mom.y), float(mom.z), mass)
    except AttributeError:
        p4.SetXYZM(float(mom.X()), float(mom.Y()), float(mom.Z()), mass)
    return p4


def is_final_state_mc(p):
    """
    Heurística robusta para identificar estado final estable en MCParticles.
    - Si existe getGeneratorStatus(): status == 1 suele indicar estable.
    - Si existe getDaughters(): sin hijas suele ser final.
    """
    try:
        st = p.getGeneratorStatus()
        if int(st) == 1:
            return True
    except Exception:
        pass

    try:
        d = p.getDaughters()
        # En EDM4hep, si no tiene hijas normalmente es final
        if len(d) == 0:
            return True
    except Exception:
        pass

    # Fallback conservador
    return False


def categorize_pdg(pdg):
    """
    Clasificación simple por PDG.
    Devuelve una etiqueta entre:
    photon, electron, muon, charged_hadron, neutral_hadron, other
    """
    apdg = abs(int(pdg))
    if apdg == 22:
        return "photon"
    if apdg == 11:
        return "electron"
    if apdg == 13:
        return "muon"

    # Hadrones cargados típicos (pi±, K±, p±, etc.)
    if apdg in (211, 321, 2212, 3112, 3222, 3312, 3334):
        return "charged_hadron"

    # Hadrones neutros típicos (n, K0L, K0S, Lambda0, pi0...)
    if apdg in (2112, 130, 310, 3122, 111):
        return "neutral_hadron"

    return "other"


# ----------------------------------------------------------------------------
def my_hook(parser):
    parser.add_argument(
        "--sys-err",
        type=str,
        default="config/systematics/err_sys.yml",
        help="YAML file with systematics errors to apply",
    )
    parser.add_argument(
        "--test-extremes",
        action="store_true",
        help="Test the extremes of the photon energy resolution",
    )


# Load config (necessary for set up the logger)
default_config = "config/default/taurecolong.yaml"
outputbasepath = "Results/TauRecoZqq/"

general_configs = myutils.setup_analysis_config(
    default_config, outputbasepath, parser_hook=my_hook
)

loggers = general_configs["loggers"]
run_config = general_configs["config"]
args = general_configs["args"]

logger_config = loggers["config"]
logger_io = loggers["io"]
logger_process = loggers["processing"]

test_pfo = args.test_pfo

# Cuts
dRMax = run_config["cuts"]["dRMax"]
minPTauPhoton = run_config["cuts"]["TauPhotonPCut"]
minPTauPion = run_config["cuts"]["TauPionPCut"]
PNeutron = run_config["cuts"]["NeutronCut"]
generalPCut = run_config["cuts"]["generalPCut"]

# Output
outputpath = general_configs["outputpath"]
fileOutName = os.path.join(outputpath, general_configs["fileOutName"])

# Systematics config
sys_errors = general_configs["config"].get("systematics_errors", {})
photon_config = sys_errors.get("photon_config", {})
test_extremes = args.test_extremes

# General configuration
sample = run_config["general"]["sample"]
test_arg = general_configs["flags"]["test"]

logger_config.info("Configuration loaded!")
logger_config.info("Configuration:\n%s", pprint.pformat(general_configs, indent=4))

# I/O
gatr_results_path = general_configs["args"].gatr_result
filenames, mlpf_results = myutils.get_root_trees_path(
    sample, gatr_results_path, loggers, test_arg
)
reader = root_io.Reader(filenames)
logger_io.info("Read %d files", len(filenames))
logger_io.info("First %s files.", filenames[:10])

# Collections
genparts = "MCParticles"
pfobjects = "PandoraPFOs"

# ----------------------------------------------------------------------------
# Histograms from YAML
histogram_config = general_configs.get("histograms_config", {})
root_histograms = myutils.set_up_root_histograms(histogram_config)

if test_extremes:
    logger_process.info("Testing extremes is enabled.")
    root_histograms_super = {
        "original": root_histograms,
        "min_err": myutils.clone_histograms_with_suffix(root_histograms, "_min"),
        "max_err": myutils.clone_histograms_with_suffix(root_histograms, "_max"),
    }
else:
    root_histograms_super = {"original": root_histograms}

# ----------------------------------------------------------------------------
# Output tables (reco-only)
reco_summary = {
    "Event": [],
    "ExtremesKey": [],
    "NCandidatesTotal": [],
    "NRecoTaus": [],
    "NRecoElectrons": [],
    "NRecoMuons": [],
}

# detalle por candidato (opcional, útil para depurar “ruido tau”)
reco_candidates_rows = {
    "Event": [],
    "ExtremesKey": [],
    "CandidateIndex": [],
    "CandidateKind": [],   # "tau"/"electron"/"muon"
    "RecoID": [],
    "Charge": [],
    "P": [],
    "Pt": [],
    "Eta": [],
    "Theta": [],
    "NConsts": [],
    "NPhotons": [],
    "NPions": [],
    "NNeutrons": [],
}

# ----------------------------------------------------------------------------
countEvents = 0

for eventid, event in enumerate(reader.get("events")):
    logger_process.debug("Processing event %d", eventid)
    if countEvents % 1000 == 0:
        logger_process.info("Processing event %d", countEvents)
    countEvents += 1

    mc_particles = event.get(genparts)
    pfos = event.get(pfobjects)

    # -------------------------
    # GEN: SOLO info de partículas finales (si quieres histos a ese nivel)
    #      NO taus, NO tau-decay parsing.
    # -------------------------
    # Rellenamos histogramas SOLO si existen en tu YAML (no asumimos nombres).
    # Sugerencia: crea en YAML histos como:
    #   GenFinal/Events/PhotonP, ElectronP, MuonP, ChargedHadronP, NeutralHadronP, ...
    # y aquí los rellenará.
    for p in mc_particles:
        if not is_final_state_mc(p):
            continue
        try:
            pdg = int(p.getPDG())
        except Exception:
            continue
        cat = categorize_pdg(pdg)
        p4 = p4_from_particle(p)

        # Ejemplos de rutas (no obligatorias): se rellenan si existen
        safe_fill(root_histograms_super["original"], ("GenFinal", "Events", f"{cat.capitalize()}P"), p4.P())
        safe_fill(root_histograms_super["original"], ("GenFinal", "Events", f"{cat.capitalize()}Theta"), p4.Theta())
        safe_fill(root_histograms_super["original"], ("GenFinal", "Events", "AllFinalP"), p4.P())
        safe_fill(root_histograms_super["original"], ("GenFinal", "Events", "AllFinalTheta"), p4.Theta())

    # -------------------------
    # RECO: reconstrucción (taus/e/mu) con tu pipeline existente
    # -------------------------
    recoTau, recoElectrons, recoMuons, recoTau_max, recoTau_min = extractTauDecays(
        gatr_results_path,
        mlpf_results,
        eventid,
        pfos,
        dRMax,
        minPTauPhoton,
        minPTauPion,
        PNeutron,
        generalPCut,
        photon_config,
        test_extremes,
        test_pfo,
        logger_process,
    )

    recoTaus_extremes = {
        "original": recoTau,
        "min_err": recoTau_min,
        "max_err": recoTau_max,
    } if test_extremes else {"original": recoTau}

    # Para extremes: procesamos igual que antes, pero SIN matching con gen
    for key in root_histograms_super.keys():
        logger_process.debug("Processing extremes type: %s", key)

        root_histograms_k = root_histograms_super[key]
        recoTau_k = recoTaus_extremes.get(key, recoTau)

        nRecoTaus = len(recoTau_k)
        nRecoElectrons = len(recoElectrons)
        nRecoMuons = len(recoMuons)
        
        # Construimos lista unificada de candidatos (igual espíritu que el base)
        candidates = []
        for t_i, t in recoTau_k.items():
            candidates.append(("tau", t))
        for e_i, e in recoElectrons.items():
            candidates.append(("tau", e))
        for m_i, m in recoMuons.items():
            candidates.append(("tau", m))

        reco_summary["Event"].append(eventid)
        reco_summary["ExtremesKey"].append(key)
        reco_summary["NCandidatesTotal"].append(len(candidates))
        reco_summary["NRecoTaus"].append(nRecoTaus)
        reco_summary["NRecoElectrons"].append(nRecoElectrons)
        reco_summary["NRecoMuons"].append(nRecoMuons)
        if not (
                nRecoTaus == 1 and
                ((nRecoElectrons == 1) ^ (nRecoMuons == 1))
            ):
          continue # Ignoring this event to get the bk
        cand1_p4, cand2_p4 = None, None
        if nRecoTaus == 1:
            cand1_p4 = recoTau_k[0].getMomentum()
        if nRecoElectrons == 1:
            cand2_p4 = recoElectrons[0].getMomentum()
        elif nRecoMuons == 1:
            cand2_p4 = recoMuons[0].getMomentum()
        print(nRecoElectrons, nRecoMuons, nRecoTaus)
        dR_between = myutils.dRAngle(cand1_p4, cand2_p4)
        if dR_between < 1:
            continue # Ignoring this event to get the bk
        # Histos reco: si existen en YAML, se rellenan; si no, no pasa nada.
        safe_fill(root_histograms_k, ("Reco", "Events", "NTaus"), nRecoTaus)

        # Detalle por candidato y algunos histos “genéricos”
        for idx, (kind, cand) in enumerate(candidates):
            try:
                p4 = cand.getMomentum()
            except Exception:
                continue

            # asegurar TLorentzVector
            if not isinstance(p4, ROOT.TLorentzVector):
                p4_ = ROOT.TLorentzVector()
                try:
                    p4_.SetXYZM(p4.x, p4.y, p4.z, cand.getMass())
                except Exception:
                    try:
                        p4_.SetXYZM(p4.X(), p4.Y(), p4.Z(), cand.getMass())
                    except Exception:
                        continue
                p4 = p4_

            try:
                rid = int(cand.getID())
            except Exception:
                rid = -999
            try:
                q = float(cand.getCharge())
            except Exception:
                q = 0.0
            try:
                nconst = int(cand.getnConst())
                daughters = cand.getDaughters()
            except Exception:
                nconst = 0
                daughters = []

            # contadores (útiles para tu “ruido”)
            n_ph = 0
            n_pi = 0
            n_n = 0
            for c in range(0, nconst):
                const = daughters[c]
                try:
                    pdg = int(const.getPDG())
                except Exception:
                    continue
                apdg = abs(pdg)
                if apdg == 22:
                    n_ph += 1
                elif apdg == 211:
                    n_pi += 1
                elif apdg == 2112:
                    n_n += 1

            # Tabla detalle
            reco_candidates_rows["Event"].append(eventid)
            reco_candidates_rows["ExtremesKey"].append(key)
            reco_candidates_rows["CandidateIndex"].append(idx)
            reco_candidates_rows["CandidateKind"].append(kind)
            reco_candidates_rows["RecoID"].append(rid)
            reco_candidates_rows["Charge"].append(q)
            reco_candidates_rows["P"].append(p4.P())
            reco_candidates_rows["Pt"].append(p4.Pt())
            reco_candidates_rows["Eta"].append(p4.Eta())
            reco_candidates_rows["Theta"].append(p4.Theta())
            reco_candidates_rows["NConsts"].append(nconst)
            reco_candidates_rows["NPhotons"].append(n_ph)
            reco_candidates_rows["NPions"].append(n_pi)
            reco_candidates_rows["NNeutrons"].append(n_n)

            # Histos reco por tipo (si están definidos en YAML)
            # Ejemplos recomendados:
            #   Reco/Events/TauP, ElectronP, MuonP, TauPt, ...
            if kind == "tau":
                safe_fill(root_histograms_k, ("Reco", "Events", "TauP"), p4.P())
                safe_fill(root_histograms_k, ("Reco", "Events", "TauPt"), p4.Pt())
                safe_fill(root_histograms_k, ("Reco", "Events", "TauTheta"), p4.Theta())
                safe_fill(root_histograms_k, ("Reco", "Events", "TauEta"), p4.Eta())
                safe_fill(root_histograms_k, ("Reco", "Events", "TauType"), rid)
            elif kind == "electron":
                safe_fill(root_histograms_k, ("Reco", "Events", "ElectronP"), p4.P())
                safe_fill(root_histograms_k, ("Reco", "Events", "ElectronPt"), p4.Pt())
            elif kind == "muon":
                safe_fill(root_histograms_k, ("Reco", "Events", "MuonP"), p4.P())
                safe_fill(root_histograms_k, ("Reco", "Events", "MuonPt"), p4.Pt())

            # Conteos de constituyentes (útiles para calibrar “ruido”)
            safe_fill(root_histograms_k, ("Reco", "Events", "TauCandNPhotons"), n_ph)
            safe_fill(root_histograms_k, ("Reco", "Events", "TauCandNPions"), n_pi)
            safe_fill(root_histograms_k, ("Reco", "Events", "TauCandNNeutrons"), n_n)

# ----------------------------------------------------------------------------
logger_io.info("Processed %d events", countEvents)

# Escritura ROOT
outfile = ROOT.TFile(fileOutName, "RECREATE")

# Guardamos histogramas + eficiencias (solo donde aplique según tu YAML/config)
for key in root_histograms_super:
    rh = root_histograms_super[key]
    suffix = "" if key == "original" else f"_{key}"

    myutils.write_plot_config(rh, outputpath, suffix)
    rh = myutils.calc_efficiency(rh, histogram_config, suffix)
    write_histograms_recursive(rh)

# CSVs reco-only
reco_summary_df = pd.DataFrame(reco_summary)
reco_candidates_df = pd.DataFrame(reco_candidates_rows)

reco_summary_file = os.path.join(outputpath, "reco_summary.csv")
reco_candidates_file = os.path.join(outputpath, "reco_candidates.csv")

reco_summary_df.to_csv(reco_summary_file, index=False)
reco_candidates_df.to_csv(reco_candidates_file, index=False)

# Guardar config efectiva
output_config_file = os.path.join(outputpath, "config.yaml")
with open(output_config_file, "w") as f:
    yaml.dump(run_config, f)
    logger_io.info("Configuration file saved to %s", output_config_file)

logger_io.info("Output ROOT file %s", fileOutName)
logger_io.info("Reco summary CSV %s", reco_summary_file)
logger_io.info("Reco candidates CSV %s", reco_candidates_file)
logger_io.info("End of job")
outfile.Close()
