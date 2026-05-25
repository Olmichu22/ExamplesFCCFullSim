#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RhoHistFromTree_MDecs_parallel.py

Versión MDecs de rhoHistFromTree_parallel.py.

Lee árboles con variables tau1_*/tau2_* (generados por genOnlyRHOTree_parallel.py)
y genera histogramas para pares de desintegraciones configurables (decay_pair).

Las categorías usan doble sufijo {cat_este}_{cat_otro} por hemisferio:
  - Primer sufijo: estado del hemisferio al que pertenece la variable
  - Segundo sufijo: estado del hemisferio opuesto
La clasificación usa decayID para histogramas Gen/Matched, recoTauID para Reco.

Los histogramas se nombran con _dec0 / _dec1 según el hemisferio.
Los histogramas compartidos (ZMass, cross-hemisphere) no llevan sufijo de hemisferio.

Requiere en el YAML de config:
  general:
    decay_pair: [decID_0, decID_1]   # e.g. [2, -13] para rho + muon

USO:
    python RhoAnalysis/RhoHistFromTree_MDecs_parallel.py \\
        --tree-file Results/RhoAnalysis/.../tau_trainedAll_*.root \\
        -c config/default/taurecolong_mdecs.yaml \\
        --hist-config-mdecs config/histograms/rho_analysis_config_mdecs.yml \\
        -d -777 -v --n-workers 4
"""
import logging
import math
import multiprocessing
import numpy as np
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import yaml
import ROOT
from ROOT import TFile

from modules import myutils
from RhoAnalysis.temp_functions import extract_scalars_optional, make_p4

# ── Constantes ────────────────────────────────────────────────────────────────

_DEFAULT_CONFIG = "config/default/taurecolong.yaml"
_OUTPUT_BASE    = "Results/RhoAnalysis/"

_SHARED_MDECS = [
    ("GenZMass",    "GenZMass",    float),
    ("GenZVisMass", "GenZVisMass", float),
    ("ZRecoMass",   "ZMass",       float),
    ("beamE",       "beamE",       float),
]

_TAU_KEYS_MDECS = [
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

# Mapa de decayID a nombre de categoría BG
_BG_MAP = {-13: "BGMuon", -11: "BGEle", 0: "BGPion", 1: "BGRho", 10: "BGA1"}

# ── Pesos por hemisferio ───────────────────────────────────────────────────────
# Firma: (dec_vars, other_vars, base_weight) → effective_weight
# dec_vars: vars del hemisferio cuya variable estamos llenando
# other_vars: vars del otro hemisferio

WEIGHT_VALUES_MDECS = {
    "nominal":  lambda dv, ov, w: w,
    "P1":       lambda dv, ov, w: w * dv.get("weight_P1", 1.0),
    "M1":       lambda dv, ov, w: w * dv.get("weight_M1", 1.0),
    "corr_P1":  lambda dv, ov, w: w * dv.get("weight_P1", 1.0) * ov.get("weight_P1", 1.0),
    "corr_M1":  lambda dv, ov, w: w * dv.get("weight_M1", 1.0) * ov.get("weight_M1", 1.0),
    "other_P1": lambda dv, ov, w: w * ov.get("weight_P1", 1.0),
    "other_M1": lambda dv, ov, w: w * ov.get("weight_M1", 1.0),
}

# ── Reglas de llenado por hemisferio ──────────────────────────────────────────
# Formato: (level, base_name, x_fn(v, sh), y_fn(v, sh) | None, [cond_fn(v, sh)])
# v = vars del hemisferio, sh = shared_vars
# El nombre real en el YAML es base_name + "_dec0" o "_dec1".

FILL_RULES_PER_DEC = [
    # ── Gen ───────────────────────────────────────────────────────────────────
    ("Gen",     "Omega",              lambda v, sh: v["omega"],                    None),
    ("Gen",     "CosTheta_GEN",       lambda v, sh: v["cos_theta"],               None),
    ("Gen",     "CosPsi_GEN",         lambda v, sh: v["cos_psi"],                 None),
    ("Gen",     "CosThetaTau_GEN",    lambda v, sh: v["cos_theta_tau"],           None),
    ("Gen",     "CosThetaMeson_GEN",  lambda v, sh: math.cos(v["visTheta"]),      None),
    ("Gen",     "CosThetaMeson_Reco", lambda v, sh: math.cos(v["recoVisTheta"]),  None),
    ("Gen",     "DecayType",          lambda v, sh: v["decayID"],                 None),
    ("Gen",     "OptimalVar",         lambda v, sh: v["optimalVar"],              None),
    ("Gen",     "VisP",               lambda v, sh: v["visP"],                    None),
    ("Gen",     "TauP",               lambda v, sh: v["P"],                       None),
    ("Gen",     "Omega_ZGenMass",     lambda v, sh: v["omega"],                   lambda v, sh: sh["GenZMass"]),
    ("Gen",     "Omega_ZVisMass",     lambda v, sh: v["omega"],                   lambda v, sh: sh["GenZVisMass"]),
    # Condicionados: solo cuando omega < 0
    ("Gen",     "VisP_omega_neg",     lambda v, sh: v["visP"],                    None,
                                      lambda v, sh: v["omega"] < 0),
    ("Gen",     "TauP_omega_neg",     lambda v, sh: v["P"],                       None,
                                      lambda v, sh: v["omega"] < 0),
    ("Gen",     "TauTheta_omega_neg", lambda v, sh: v["Theta"],                   None,
                                      lambda v, sh: v["omega"] < 0),
    ("Gen",     "TauPhi_omega_neg",   lambda v, sh: v["Phi"],                     None,
                                      lambda v, sh: v["omega"] < 0),
    # ── Reco ──────────────────────────────────────────────────────────────────
    ("Reco",    "Omega_Reco",         lambda v, sh: v["omega"],                    None),
    ("Reco",    "CosTheta",           lambda v, sh: v["cos_theta"],               None),
    ("Reco",    "CosPsi",             lambda v, sh: v["cos_psi"],                 None),
    ("Reco",    "RecoVisEOverBeamE",  lambda v, sh: v["recoVisE"] / sh["beamE"]
                                                    if sh["beamE"] else 0.0,      None),
    ("Reco",    "RecoVisCosTheta",    lambda v, sh: math.cos(v["recoVisTheta"]),  None),
    ("Reco",    "RecoVisP",           lambda v, sh: v["recoVisP"],                None),
    ("Reco",    "RecoDecayType",      lambda v, sh: float(v["recoTauID"]),        None),
    ("Reco",    "Optimal_X",          lambda v, sh: v.get("_optimal_x", 0.0),    None),
    # ── Matched ───────────────────────────────────────────────────────────────
    ("Matched", "VisEOverBeamE",      lambda v, sh: v["visE"] / sh["beamE"]
                                                    if sh["beamE"] else 0.0,      None),
    ("Matched", "VisCosTheta",        lambda v, sh: math.cos(v["visTheta"]),      None),
]

# Reglas para histogramas compartidos (no por hemisferio).
# Formato: (level, hist_name, x_fn(v0, v1, sh), y_fn(v0, v1, sh) | None)
# v0 = vars_dec0, v1 = vars_dec1, sh = shared_vars
# Categoría se calcula desde la perspectiva de dec0.

FILL_RULES_SHARED = [
    ("Gen",  "GenZMass",               lambda v0, v1, sh: sh["GenZMass"],    None),
    ("Gen",  "GenVisZMass",            lambda v0, v1, sh: sh["GenZVisMass"], None),
    ("Reco", "RecoZMass",              lambda v0, v1, sh: sh["ZRecoMass"],   None),
    ("Reco", "DeltaR_dec0_dec1",       None,                                 None),  # calculado inline
    # Cross-hemisphere 2D
    ("Gen",  "VisP_dec0_vs_dec1",      lambda v0, v1, sh: v0["visP"],        lambda v0, v1, sh: v1["visP"]),
    ("Gen",  "Omega_dec0_vs_VisP_dec1",lambda v0, v1, sh: v0["omega"],       lambda v0, v1, sh: v1["visP"]),
    ("Gen",  "Omega_dec0_vs_Omega_dec1",  lambda v0, v1, sh: v0["omega"],        lambda v0, v1, sh: v1["omega"]),
    ("Gen",  "WeightP1_dec0_vs_dec1",    lambda v0, v1, sh: v0["weight_P1"],    lambda v0, v1, sh: v1["weight_P1"]),
    ("Gen",  "WeightM1_dec0_vs_dec1",    lambda v0, v1, sh: v0["weight_M1"],    lambda v0, v1, sh: v1["weight_M1"]),
]


# ── Funciones auxiliares ───────────────────────────────────────────────────────

def write_histograms_recursive(obj):
    if isinstance(obj, dict):
        for v in obj.values():
            write_histograms_recursive(v)
    else:
        try:
            obj.Write()
        except AttributeError:
            pass


def split_entry_ranges(n_entries, n_workers):
    k, rem = divmod(n_entries, n_workers)
    ranges, start = [], 0
    for i in range(n_workers):
        end = start + k + (1 if i < rem else 0)
        if start < end:
            ranges.append((start, end))
        start = end
    return ranges


def _reattach_histograms(nested, tfile):
    if isinstance(nested, dict):
        for v in nested.values():
            _reattach_histograms(v, tfile)
    elif isinstance(nested, ROOT.TH1):
        nested.SetDirectory(tfile)


def _flatten_histograms(nested, result=None):
    if result is None:
        result = {}
    if isinstance(nested, dict):
        for v in nested.values():
            _flatten_histograms(v, result)
    else:
        try:
            result[nested.GetName()] = nested
        except AttributeError:
            pass
    return result


def _assign_hemispheres(tau1_vars, tau2_vars, id0_gen, id1_gen):
    """Asigna tau1/tau2 a dec0/dec1 por decayID (order-independent).

    Devuelve (vars_dec0, vars_dec1) o (None, None) si el evento no encaja con
    el par de desintegraciones esperado.
    Para par simétrico (id0_gen == id1_gen), asigna tau1→dec0, tau2→dec1.
    """
    t1_id = int(tau1_vars.get("decayID", -999))
    t2_id = int(tau2_vars.get("decayID", -999))

    if t1_id == id0_gen and t2_id == id1_gen:
        return tau1_vars, tau2_vars
    if t2_id == id0_gen and t1_id == id1_gen:
        return tau2_vars, tau1_vars
    return None, None


def _classify_hemisphere(dec_vars, expected_gen_id, use_reco=False):
    """Devuelve la categoría de un hemisferio.

    Returns: una de 'SIGNAL', 'BGMuon', 'BGEle', 'BGPion', 'BGRho', 'BGA1', 'BGOther'
    """
    actual = int(dec_vars.get("recoTauID", -999) if use_reco else dec_vars.get("decayID", -999))
    if actual == 2:
        actual = 1  # remap reco ρ → gen ρ
    if actual == expected_gen_id:
        return "SIGNAL"
    return _BG_MAP.get(actual, "BGOther")


def _get_fill_categories(this_cat, other_cat):
    """Devuelve la lista de categorías (doble sufijo) a rellenar para este evento.

    Incluye siempre ALL_ALL.
    Incluye la combinación exacta.
    Si this_cat es un BG específico, incluye también BG_{other_cat}.
    Si other_cat es un BG específico, incluye también {this_cat}_BG.
    """
    cats = ["ALL_ALL"]
    exact = f"{this_cat}_{other_cat}"
    if exact != "ALL_ALL":
        cats.append(exact)

    this_is_bg = this_cat not in ("SIGNAL",)
    other_is_bg = other_cat not in ("SIGNAL",)

    if this_is_bg and exact != f"BG_{other_cat}":
        cats.append(f"BG_{other_cat}")
    if other_is_bg and exact != f"{this_cat}_BG":
        cats.append(f"{this_cat}_BG")
    if this_is_bg and other_is_bg:
        cats.append("BG_BG")

    return cats


def _fill_hemisphere(hists, dec_vars, other_vars, shared_vars, dec_idx,
                     this_cat_gen, other_cat_gen, this_cat_reco, other_cat_reco,
                     weight):
    """Rellena todos los histogramas de un hemisferio según FILL_RULES_PER_DEC.

    dec_idx: 0 o 1 (para construir el sufijo _dec0 / _dec1)
    La clasificación usa gen para niveles Gen/Matched y reco para Reco.
    """
    dec_suffix = f"dec{dec_idx}"

    for rule in FILL_RULES_PER_DEC:
        level, base_name, x_fn, y_fn = rule[:4]
        cond_fn = rule[4] if len(rule) > 4 else None

        if cond_fn is not None and not cond_fn(dec_vars, shared_vars):
            continue

        var_name = f"{base_name}_{dec_suffix}"

        if level not in hists or var_name not in hists[level]:
            continue

        # Clasificación por nivel
        if level == "Reco":
            this_cat, other_cat = this_cat_reco, other_cat_reco
        else:
            this_cat, other_cat = this_cat_gen, other_cat_gen

        cats_to_fill = _get_fill_categories(this_cat, other_cat)

        x_val = x_fn(dec_vars, shared_vars)
        y_val = y_fn(dec_vars, shared_vars) if y_fn is not None else None

        for cat in cats_to_fill:
            if cat not in hists[level][var_name]:
                continue
            for w_name, w_hist in hists[level][var_name][cat].items():
                eff_w = WEIGHT_VALUES_MDECS[w_name](dec_vars, other_vars, weight)
                if y_val is None:
                    w_hist.Fill(x_val, eff_w)
                else:
                    w_hist.Fill(x_val, y_val, eff_w)


def _fill_shared(hists, vars_dec0, vars_dec1, shared_vars,
                 cat_dec0_gen, cat_dec1_gen, cat_dec0_reco, cat_dec1_reco,
                 weight, dR=None):
    """Rellena histogramas compartidos según FILL_RULES_SHARED.

    La categoría se calcula desde la perspectiva de dec0.
    """
    for rule in FILL_RULES_SHARED:
        level, hist_name, x_fn, y_fn = rule[:4]
        cond_fn = rule[4] if len(rule) > 4 else None

        if cond_fn is not None and not cond_fn(vars_dec0, vars_dec1, shared_vars):
            continue

        if level not in hists or hist_name not in hists[level]:
            continue

        if level == "Reco":
            this_cat, other_cat = cat_dec0_reco, cat_dec1_reco
        else:
            this_cat, other_cat = cat_dec0_gen, cat_dec1_gen

        cats_to_fill = _get_fill_categories(this_cat, other_cat)

        # DeltaR se maneja de forma especial
        if hist_name == "DeltaR_dec0_dec1":
            if dR is None:
                continue
            for cat in cats_to_fill:
                if cat not in hists[level][hist_name]:
                    continue
                for w_name, w_hist in hists[level][hist_name][cat].items():
                    eff_w = WEIGHT_VALUES_MDECS[w_name](vars_dec0, vars_dec1, weight)
                    w_hist.Fill(dR, eff_w)
            continue

        x_val = x_fn(vars_dec0, vars_dec1, shared_vars)
        y_val = y_fn(vars_dec0, vars_dec1, shared_vars) if y_fn is not None else None

        for cat in cats_to_fill:
            if cat not in hists[level][hist_name]:
                continue
            for w_name, w_hist in hists[level][hist_name][cat].items():
                eff_w = WEIGHT_VALUES_MDECS[w_name](vars_dec0, vars_dec1, weight)
                if y_val is None:
                    w_hist.Fill(x_val, eff_w)
                else:
                    w_hist.Fill(x_val, y_val, eff_w)


def _fill_zvismassbins(hists, vars_dec0, vars_dec1, shared_vars,
                       cat_dec0_gen, cat_dec1_gen, weight):
    """Rellena los histogramas de bins de ZVisMass (lógica condicional)."""
    z_vis = shared_vars.get("GenZVisMass", 0.0)
    bin_ranges = [(0, 40, 1), (40, 70, 2), (70, 100, 3)]

    for lo, hi, idx in bin_ranges:
        if not (lo <= z_vis < hi):
            continue
        for dec_idx, dec_vars, other_vars, this_cat, other_cat in [
            (0, vars_dec0, vars_dec1, cat_dec0_gen, cat_dec1_gen),
            (1, vars_dec1, vars_dec0, cat_dec1_gen, cat_dec0_gen),
        ]:
            var_name = f"Omega_ZVisMass_Bin{idx}_dec{dec_idx}"
            if "Gen" not in hists or var_name not in hists["Gen"]:
                continue
            cats_to_fill = _get_fill_categories(this_cat, other_cat)
            for cat in cats_to_fill:
                if cat not in hists["Gen"][var_name]:
                    continue
                for w_name, w_hist in hists["Gen"][var_name][cat].items():
                    eff_w = WEIGHT_VALUES_MDECS[w_name](dec_vars, other_vars, weight)
                    w_hist.Fill(dec_vars["omega"], eff_w)
        break


# ── Función principal de llenado por rango ────────────────────────────────────

def process_tree_range_mdecs(trees, root_histograms_super,
                              weight, decay_pair,
                              cuts_cfg, logger_process, other_BG_id,
                              start_entry, end_entry):
    """Procesa entradas [start_entry, end_entry) de los árboles MDecs.

    Devuelve dict de contadores: totalEvents, selectedEvents, sumWeights, ...
    """
    id0_gen = 1 if decay_pair[0] == 2 else decay_pair[0]
    id1_gen = 1 if decay_pair[1] == 2 else decay_pair[1]

    tauPCut    = cuts_cfg.get("tauPCut",   0.0)
    meson_cut  = cuts_cfg.get("meson_cut",  [0.0, np.inf])
    lepton_cut = cuts_cfg.get("lepton_cut", [0.0, np.inf])
    zmass_cut  = cuts_cfg.get("zmass_cut",  [0.0, np.inf])
    angle_sep  = cuts_cfg.get("angle_sep",  [0.0, np.inf])
    extra_cuts = cuts_cfg.get("extra_cuts", [])

    totalEvents    = 0
    selectedEvents = 0
    sumWeights     = 0.0
    sumWeightsP1   = 0.0
    sumWeightsM1   = 0.0

    for tree_key, tree in trees.items():
        root_hists = root_histograms_super[tree_key]

        for i in range(start_entry, end_entry):
            tree.GetEntry(i)
            entry = tree

            if tree_key == "original":
                totalEvents += 1

            # Extraer ramas compartidas
            shared_vars = extract_scalars_optional(entry, _SHARED_MDECS, default=0.0)
            beamE = shared_vars["beamE"]

            # Extraer ramas por tau
            tau1_vars = {k: float(getattr(entry, f"tau1_{k}", 0.0)) for k in _TAU_KEYS_MDECS}
            tau2_vars = {k: float(getattr(entry, f"tau2_{k}", 0.0)) for k in _TAU_KEYS_MDECS}

            # Asignar hemisferios según decayID
            vars_dec0, vars_dec1 = _assign_hemispheres(tau1_vars, tau2_vars, id0_gen, id1_gen)
            if vars_dec0 is None:
                continue

            # Cálculos derivados
            for vd in (vars_dec0, vars_dec1):
                vd["_optimal_x"] = (2.0 * vd["recoVisE"] / beamE - 1.0) if beamE else 0.0

            # Cortes (dec0 = "mesón", dec1 = "leptón" por convención; siempre aplicables)
            if vars_dec0["recoVisP"] < tauPCut:
                continue
            zmass = shared_vars["ZRecoMass"]
            if not (zmass_cut[0] <= zmass <= zmass_cut[1]):
                continue
            if not (meson_cut[0] <= vars_dec0["recoVisP"] <= meson_cut[1]):
                continue
            if not (lepton_cut[0] <= vars_dec1["recoVisP"] <= lepton_cut[1]):
                continue

            p4_dec0 = make_p4(vars_dec0["recoVisP"], vars_dec0["recoVisTheta"],
                               vars_dec0["recoVisPhi"], vars_dec0["recoVisE"])
            p4_dec1 = make_p4(vars_dec1["recoVisP"], vars_dec1["recoVisTheta"],
                               vars_dec1["recoVisPhi"], vars_dec1["recoVisE"])
            dR = myutils.dRAngle(p4_dec0, p4_dec1)
            if not (angle_sep[0] <= dR <= angle_sep[1]):
                continue

            if extra_cuts:
                skip = False
                flat = {**vars_dec0, **{f"d1_{k}": v for k, v in vars_dec1.items()},
                        **shared_vars}
                for expr in extra_cuts:
                    try:
                        if not eval(expr, {"__builtins__": {}}, flat):
                            skip = True
                            break
                    except Exception:
                        skip = True
                        break
                if skip:
                    continue

            # Clasificación gen y reco por hemisferio
            cat_dec0_gen  = _classify_hemisphere(vars_dec0, id0_gen, use_reco=False)
            cat_dec1_gen  = _classify_hemisphere(vars_dec1, id1_gen, use_reco=False)
            cat_dec0_reco = _classify_hemisphere(vars_dec0, id0_gen, use_reco=True)
            cat_dec1_reco = _classify_hemisphere(vars_dec1, id1_gen, use_reco=True)

            if tree_key == "original":
                selectedEvents += 1
                if cat_dec0_gen == "SIGNAL" and cat_dec1_gen == "SIGNAL":
                    sumWeights   += weight
                    sumWeightsP1 += weight * vars_dec0.get("weight_P1", 1.0)
                    sumWeightsM1 += weight * vars_dec0.get("weight_M1", 1.0)
                else:
                    other_BG_id[f"{cat_dec0_gen}_{cat_dec1_gen}"] = (
                        other_BG_id.get(f"{cat_dec0_gen}_{cat_dec1_gen}", 0) + 1)

            # Rellenar histogramas por hemisferio
            _fill_hemisphere(root_hists, vars_dec0, vars_dec1, shared_vars, 0,
                             cat_dec0_gen, cat_dec1_gen, cat_dec0_reco, cat_dec1_reco, weight)
            _fill_hemisphere(root_hists, vars_dec1, vars_dec0, shared_vars, 1,
                             cat_dec1_gen, cat_dec0_gen, cat_dec1_reco, cat_dec0_reco, weight)

            # Rellenar histogramas compartidos
            _fill_shared(root_hists, vars_dec0, vars_dec1, shared_vars,
                         cat_dec0_gen, cat_dec1_gen, cat_dec0_reco, cat_dec1_reco,
                         weight, dR=dR)

            # ZVisMass bins (lógica condicional inline)
            _fill_zvismassbins(root_hists, vars_dec0, vars_dec1, shared_vars,
                               cat_dec0_gen, cat_dec1_gen, weight)

    return {
        "totalEvents":    totalEvents,
        "selectedEvents": selectedEvents,
        "sumWeights":     sumWeights,
        "sumWeightsP1":   sumWeightsP1,
        "sumWeightsM1":   sumWeightsM1,
    }


# ── Worker ────────────────────────────────────────────────────────────────────

def process_chunk_stage2_mdecs(input_root, tree_keys, entry_range, config_bundle, worker_id):
    """Worker: procesa un rango de entradas y escribe histogramas parciales."""
    outputpath = config_bundle["outputpath"]

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
    logger = logging.getLogger(f"worker_{worker_id}")
    logger.info("Worker %d: entradas [%d, %d)", worker_id, entry_range[0], entry_range[1])

    hist_config      = dict(config_bundle["hist_config"])
    hist_config.pop("_all_cats", None)
    decay_pair       = config_bundle["decay_pair"]
    weight           = config_bundle["weight"]
    cuts_cfg         = config_bundle["cuts_cfg"]
    fileOutName_base = config_bundle["fileOutName_base"]

    infile = TFile.Open(input_root, "READ")
    if not infile or infile.IsZombie():
        raise RuntimeError(f"Worker {worker_id}: no se puede abrir {input_root}")

    trees = {}
    for key in tree_keys:
        t = infile.Get(f"outtree_{key}")
        if isinstance(t, ROOT.TTree):
            trees[key] = t
        else:
            logger.warning("Worker %d: árbol 'outtree_%s' no encontrado", worker_id, key)

    partial_path = os.path.join(
        outputpath, f"partial_histos_worker{worker_id}_{fileOutName_base}.root")
    pfile = TFile(partial_path, "RECREATE")
    pfile.cd()

    base_hists = myutils.build_histogram_registry(hist_config)
    root_histograms_super = {"original": base_hists}

    if "min_err" in trees:
        root_histograms_super["min_err"] = myutils.clone_histograms_with_suffix(base_hists, "_min")
        _reattach_histograms(root_histograms_super["min_err"], pfile)
    if "max_err" in trees:
        root_histograms_super["max_err"] = myutils.clone_histograms_with_suffix(base_hists, "_max")
        _reattach_histograms(root_histograms_super["max_err"], pfile)

    other_BG_id = {}
    start_entry, end_entry = entry_range
    counters = process_tree_range_mdecs(
        trees=trees,
        root_histograms_super=root_histograms_super,
        weight=weight,
        decay_pair=decay_pair,
        cuts_cfg=cuts_cfg,
        logger_process=logger,
        other_BG_id=other_BG_id,
        start_entry=start_entry,
        end_entry=end_entry,
    )

    infile.Close()

    pfile.cd()
    for tree_key in root_histograms_super:
        write_histograms_recursive(root_histograms_super[tree_key])
    pfile.Close()

    logger.info("Worker %d: terminado. Events=%d Selected=%d",
                worker_id, counters["totalEvents"], counters["selectedEvents"])
    return partial_path, counters, other_BG_id


# ── Merge ─────────────────────────────────────────────────────────────────────

def merge_histogram_dicts_from_files(partial_files, root_histograms_super):
    all_main = {}
    for subtree in root_histograms_super.values():
        _flatten_histograms(subtree, all_main)

    for partial_path in partial_files:
        pfile = TFile.Open(partial_path, "READ")
        if not pfile or pfile.IsZombie():
            print(f"[WARN] No se puede abrir parcial {partial_path}")
            continue
        for name, main_hist in all_main.items():
            partial_hist = pfile.Get(name)
            if partial_hist and isinstance(partial_hist, ROOT.TH1):
                main_hist.Add(partial_hist)
        pfile.Close()


# ── Hook y main ───────────────────────────────────────────────────────────────

def my_hook(parser):
    parser.add_argument("--tree-file", type=str, required=True,
        help="Input ROOT file con outtree_original (y opcionalmente outtree_min_err/max_err)")
    parser.add_argument("--ang", type=float, default=[0.0, np.inf], nargs="+",
        help="Separación angular entre decays (por defecto: 0.0 a infinito)")
    parser.add_argument("--meson-cut", type=float, default=[0.0, np.inf], nargs="+",
        help="Rango de corte en P del dec0 (por defecto: sin corte)")
    parser.add_argument("--lepton-cut", type=float, default=[0.0, np.inf], nargs="+",
        help="Rango de corte en P del dec1 (por defecto: sin corte)")
    parser.add_argument("--zmass-cut", type=float, default=[0.0, np.inf], nargs="+",
        help="Rango de masa Z reco (por defecto: sin corte)")
    parser.add_argument("--hist-config-mdecs", type=str,
        default="config/histograms/rho_analysis_config_mdecs.yml",
        help="Config YAML de histogramas MDecs")
    parser.add_argument("--n-workers", type=int, default=None,
        help="Número de workers paralelos (default: min(n_entries, n_cpus))")
    parser.add_argument("--cut", type=str, nargs="+", default=[], metavar="EXPR",
        help="Expresiones Python extra sobre vars_dict (vars de dec0 + d1_* para dec1)")
    parser.add_argument("--decay-pair", type=int, nargs=2, default=None,
        metavar=("DECID_0", "DECID_1"),
        help="Par de decayIDs a analizar (sobrescribe general.decay_pair del YAML). "
             "Ejemplo: --decay-pair 2 -13")


def main():
    general_configs = myutils.setup_analysis_config(
        _DEFAULT_CONFIG, _OUTPUT_BASE, parser_hook=my_hook)
    loggers    = general_configs["loggers"]
    run_config = general_configs["config"]
    args       = general_configs["args"]
    logger_config  = loggers["config"]
    logger_io      = loggers["io"]
    logger_process = loggers["processing"]

    # Leer decay_pair: CLI tiene prioridad sobre el YAML
    if args.decay_pair is not None:
        decay_pair = args.decay_pair
    else:
        decay_pair = run_config.get("general", {}).get("decay_pair")
    if decay_pair is None or len(decay_pair) != 2:
        logger_io.error(
            "Especifica el par de desintegraciones con --decay-pair DECID_0 DECID_1 "
            "o con 'general.decay_pair: [decID_0, decID_1]' en el YAML de config."
        )
        sys.exit(1)
    decay_pair = [int(d) for d in decay_pair]
    logger_config.info("decay_pair: %s", decay_pair)

    tauPCut = run_config["cuts"]["tauCut"]

    angle_sep  = args.ang;        angle_sep  = [angle_sep[0],  np.inf] if len(angle_sep)  == 1 else angle_sep
    meson_cut  = args.meson_cut;  meson_cut  = [meson_cut[0],  np.inf] if len(meson_cut)  == 1 else meson_cut
    lepton_cut = args.lepton_cut; lepton_cut = [lepton_cut[0], np.inf] if len(lepton_cut) == 1 else lepton_cut
    zmass_cut  = args.zmass_cut;  zmass_cut  = [zmass_cut[0],  np.inf] if len(zmass_cut)  == 1 else zmass_cut

    input_root = args.tree_file
    if not os.path.isfile(input_root):
        logger_io.error("Input ROOT file %s not found", input_root)
        sys.exit(1)

    # Cargar config de histogramas
    with open(args.hist_config_mdecs, "r") as f:
        hist_config = yaml.safe_load(f)
    # Eliminar anclas YAML auxiliares que no son secciones de histogramas
    hist_config.pop("_all_cats", None)

    # Detectar árboles y número de entradas
    infile = TFile.Open(input_root, "READ")
    if not infile or infile.IsZombie():
        logger_io.error("Could not open %s", input_root)
        sys.exit(1)
    tree_keys = []
    for key in ["original", "min_err", "max_err"]:
        t = infile.Get(f"outtree_{key}")
        if isinstance(t, ROOT.TTree) and t.GetEntries() > 0:
            tree_keys.append(key)
            logger_io.info("Found tree outtree_%s with %d entries", key, t.GetEntries())
    if "original" not in tree_keys:
        logger_io.error("outtree_original not found or empty in %s", input_root)
        sys.exit(1)
    n_entries = infile.Get("outtree_original").GetEntries()
    infile.Close()

    weight = 1.0
    cuts_cfg = {
        "tauPCut":   tauPCut,
        "meson_cut": meson_cut,
        "lepton_cut": lepton_cut,
        "zmass_cut": zmass_cut,
        "angle_sep": angle_sep,
        "extra_cuts": args.cut,
    }

    outputpath = os.path.dirname(input_root)
    out_prefix = f"HistosMDecs_{decay_pair[0]}_{decay_pair[1]}_"
    if angle_sep[0] > 0:
        out_prefix += f"dRgt{angle_sep[0]}_{angle_sep[1]}_"
    if meson_cut[0] > 0 or meson_cut[1] < 100:
        out_prefix += f"Dec0Pgt{meson_cut[0]}_lt{meson_cut[1]}_"
    if lepton_cut[0] > 0 or lepton_cut[1] < 100:
        out_prefix += f"Dec1Pgt{lepton_cut[0]}_lt{lepton_cut[1]}_"
    if zmass_cut[0] > 0 or zmass_cut[1] < 200:
        out_prefix += f"Zmassgt{zmass_cut[0]}_lt{zmass_cut[1]}_"
    if args.cut:
        safe = "_".join(e.replace(" ", "").replace("==", "eq").replace(">", "gt").replace("<", "lt")
                        for e in args.cut)
        out_prefix += f"cut_{safe}_"
    fileOutName = os.path.join(outputpath, out_prefix + general_configs["fileOutName"])
    fileOutName_base = Path(fileOutName).stem
    os.makedirs(outputpath, exist_ok=True)

    n_workers = args.n_workers or min(n_entries, os.cpu_count() or 1)
    logger_io.info("n_entries=%d, n_workers=%d", n_entries, n_workers)

    # Modo secuencial
    if n_workers == 1:
        logger_io.info("Sequential mode.")
        base_hists = myutils.build_histogram_registry(hist_config)
        root_histograms_super = {"original": base_hists}

        infile = TFile.Open(input_root, "READ")
        trees = {}
        for key in tree_keys:
            t = infile.Get(f"outtree_{key}")
            if isinstance(t, ROOT.TTree):
                trees[key] = t
        if "min_err" in trees:
            root_histograms_super["min_err"] = myutils.clone_histograms_with_suffix(base_hists, "_min")
        if "max_err" in trees:
            root_histograms_super["max_err"] = myutils.clone_histograms_with_suffix(base_hists, "_max")

        other_BG_id = {}
        counters = process_tree_range_mdecs(
            trees=trees,
            root_histograms_super=root_histograms_super,
            weight=weight,
            decay_pair=decay_pair,
            cuts_cfg=cuts_cfg,
            logger_process=logger_process,
            other_BG_id=other_BG_id,
            start_entry=0,
            end_entry=n_entries,
        )
        infile.Close()
        _write_output_mdecs(fileOutName, root_histograms_super, counters, other_BG_id, logger_io)
        return

    # Modo paralelo
    entry_ranges = split_entry_ranges(n_entries, n_workers)
    n_chunks     = len(entry_ranges)

    config_bundle = {
        "hist_config":      hist_config,
        "decay_pair":       decay_pair,
        "weight":           weight,
        "cuts_cfg":         cuts_cfg,
        "outputpath":       outputpath,
        "fileOutName_base": fileOutName_base,
    }

    ctx = multiprocessing.get_context("fork")
    partial_files = []
    all_counters  = []
    all_bg_ids    = []
    t_start = time.time()

    logger_io.info("Launching %d workers...", n_chunks)
    with ProcessPoolExecutor(max_workers=n_chunks, mp_context=ctx) as executor:
        futures = {
            executor.submit(
                process_chunk_stage2_mdecs,
                input_root, tree_keys, entry_ranges[i], config_bundle, i,
            ): i
            for i in range(n_chunks)
        }
        for n_done, future in enumerate(as_completed(futures), start=1):
            wid = futures[future]
            try:
                partial_path, counters, bg_ids = future.result()
                partial_files.append(partial_path)
                all_counters.append(counters)
                all_bg_ids.append(bg_ids)
                elapsed = time.time() - t_start
                print(f"  [{n_done}/{n_chunks}] worker {wid} terminado "
                      f"({counters['totalEvents']} eventos, {elapsed:.1f}s)", flush=True)
            except Exception as exc:
                logger_io.error("Worker %d falló: %s", wid, exc)
                raise

    # Reconstruir histogramas vacíos y fusionar
    base_hists = myutils.build_histogram_registry(hist_config)
    root_histograms_super = {"original": base_hists}
    if "min_err" in tree_keys:
        root_histograms_super["min_err"] = myutils.clone_histograms_with_suffix(base_hists, "_min")
    if "max_err" in tree_keys:
        root_histograms_super["max_err"] = myutils.clone_histograms_with_suffix(base_hists, "_max")

    logger_io.info("Merging %d partial files...", len(partial_files))
    merge_histogram_dicts_from_files(partial_files, root_histograms_super)

    for p in partial_files:
        try:
            os.remove(p)
        except OSError:
            pass

    merged_counters = {
        "totalEvents":    sum(c["totalEvents"]    for c in all_counters),
        "selectedEvents": sum(c["selectedEvents"]  for c in all_counters),
        "sumWeights":     sum(c["sumWeights"]      for c in all_counters),
        "sumWeightsP1":   sum(c["sumWeightsP1"]    for c in all_counters),
        "sumWeightsM1":   sum(c["sumWeightsM1"]    for c in all_counters),
    }
    merged_bg_ids = {}
    for d in all_bg_ids:
        for k, v in d.items():
            merged_bg_ids[k] = merged_bg_ids.get(k, 0) + v

    logger_io.info("Total: events=%d selected=%d (%.1fs)",
                   merged_counters["totalEvents"], merged_counters["selectedEvents"],
                   time.time() - t_start)

    _write_output_mdecs(fileOutName, root_histograms_super, merged_counters, merged_bg_ids, logger_io)


def _write_output_mdecs(fileOutName, root_histograms_super, counters, other_BG_id, logger_io):
    import pandas as pd
    logger_io.info("Writing output ROOT file %s", fileOutName)
    logger_io.info("Events=%d Selected=%d", counters["totalEvents"], counters["selectedEvents"])

    outfile = TFile(fileOutName, "RECREATE")
    outfile.cd()
    for tree_key in root_histograms_super:
        write_histograms_recursive(root_histograms_super[tree_key])
    outfile.Close()

    csv_name = fileOutName.replace(".root", "_otherBGid.csv")
    df = pd.DataFrame(sorted(other_BG_id.items()), columns=["category", "count"])
    df.to_csv(csv_name, index=False)
    logger_io.info("BG category counts saved to %s", csv_name)
    logger_io.info("Done. Results in %s", fileOutName)


if __name__ == "__main__":
    main()
