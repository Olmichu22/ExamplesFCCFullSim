#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Versión paralela de rhoHistFromTree_refactor.py.

Divide el rango de entradas del árbol en chunks y los procesa en paralelo.
Cada worker escribe un ROOT parcial con histogramas; el proceso principal
los fusiona usando TH1.Add().

USO:
    python rhoHistFromTree_parallel.py [mismos args que rhoHistFromTree_refactor.py] \\
        --n-workers N
"""
import copy
import logging
import math
import multiprocessing
import numpy as np
import os
import pprint
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import yaml
import ROOT
from ROOT import TFile

from modules import myutils
from modules import optimalVariabRho
from RhoAnalysis.temp_functions import (
    FILL_RULES, WEIGHT_VALUES,
    fill_category, fill_special_signal_histograms,
    SCALAR_BRANCHES, SCALAR_BRANCHES_WEIGHTS, SCALAR_BRANCHES_STORED_WEIGHTS,
    extract_scalars, make_p4, get_entry_vars,
)

# ── Constantes ────────────────────────────────────────────────────────────────

_DEFAULT_CONFIG = "config/default/taurecolong.yaml"
_OUTPUT_BASE    = "Results/RhoAnalysis/"


# ── Helpers (módulo-level, picklables) ────────────────────────────────────────

def write_histograms_recursive(obj):
    if isinstance(obj, dict):
        for v in obj.values():
            write_histograms_recursive(v)
    else:
        try:
            obj.Write()
        except AttributeError:
            print(f"Objeto {obj} no tiene método .Write(). Ignorado.")


def split_entry_ranges(n_entries, n_workers):
    """Divide n_entries en n_workers rangos contiguos (start, end) con end exclusivo."""
    k, rem = divmod(n_entries, n_workers)
    ranges, start = [], 0
    for i in range(n_workers):
        end = start + k + (1 if i < rem else 0)
        if start < end:
            ranges.append((start, end))
        start = end
    return ranges


def _reattach_histograms(nested, tfile):
    """Vincula todos los histogramas del dict anidado al TFile dado."""
    if isinstance(nested, dict):
        for v in nested.values():
            _reattach_histograms(v, tfile)
    elif isinstance(nested, ROOT.TH1):
        nested.SetDirectory(tfile)


def _flatten_histograms(nested, result=None):
    """Devuelve dict {name: hist_object} aplanando la estructura anidada."""
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


# ── process_tree_range: copia de process_tree con rango de entradas ───────────

def process_tree_range(rho_vars_extremes_trees,
                       root_histograms_super,
                       special_histograms_super,
                       weight,
                       selectGEN,
                       cuts_cfg,
                       proccesing_cfg,
                       logger_process,
                       other_BG_id,
                       start_entry,
                       end_entry):
    """
    Igual que temp_functions.process_tree pero sólo procesa entradas
    [start_entry, end_entry) (end_entry exclusivo).

    Definida aquí para no modificar temp_functions.py.
    """
    totalEvents   = 0
    selectedEvents = 0
    sumWeights    = 0.0
    sumWeightsP1  = 0.0
    sumWeightsM1  = 0.0

    tauPCut    = cuts_cfg.get("tauPCut", 0)
    meson_cut  = cuts_cfg.get("meson_cut", [0.0, np.inf])
    lepton_cut = cuts_cfg.get("lepton_cut", [0.0, np.inf])
    zmass_cut  = cuts_cfg.get("zmass_cut", [0.0, np.inf])
    angle_sep  = cuts_cfg.get("angle_sep", [0.0, np.inf])
    extra_cuts = cuts_cfg.get("extra_cuts", [])

    for tree_key, tree in rho_vars_extremes_trees.items():
        root_histograms = root_histograms_super[tree_key]

        for i in range(start_entry, end_entry):
            tree.GetEntry(i)
            entry = tree

            if tree_key == "original":
                totalEvents += 1

            vars_dict = get_entry_vars(entry, proccesing_cfg, logger_process)
            if vars_dict is None:
                continue

            if vars_dict["beamE"] != 0:
                vars_dict["Optimal_var_x"] = 2.0 * vars_dict["recoMesonE"] / vars_dict["beamE"] - 1.0
            else:
                vars_dict["Optimal_var_x"] = 0.0

            if vars_dict["recoMesonP"] < tauPCut:
                continue

            z_p4  = vars_dict["mesonp4"] + vars_dict["leptonp4"]
            zmass = z_p4.M()
            vars_dict["zmass"] = zmass

            if vars_dict["recoMesonP"] < meson_cut[0] or vars_dict["recoMesonP"] > meson_cut[1]:
                continue
            if vars_dict["leptonP"] < lepton_cut[0] or vars_dict["leptonP"] > lepton_cut[1]:
                continue
            if zmass < zmass_cut[0] or zmass > zmass_cut[1]:
                continue

            dR_between = myutils.dRAngle(vars_dict["mesonp4"], vars_dict["leptonp4"])
            vars_dict["dR_lep_meson"] = dR_between
            if dR_between < angle_sep[0] or dR_between > angle_sep[1]:
                continue

            vars_dict["recoMesonPt"] = vars_dict["recoMesonP"] * math.sin(vars_dict["recoMesonTheta"])

            if extra_cuts:
                skip = False
                for expr in extra_cuts:
                    try:
                        if not eval(expr, {"__builtins__": {}}, vars_dict):
                            skip = True
                            break
                    except Exception:
                        skip = True
                        break
                if skip:
                    continue

            fill_category(root_histograms, vars_dict, "ALL", weight)

            if vars_dict["genTauID"] == selectGEN:
                if tree_key == "original":
                    selectedEvents += 1
                    sumWeights     += weight
                    sumWeightsP1   += weight * vars_dict["weight_P1"]
                    sumWeightsM1   += weight * vars_dict["weight_M1"]
                fill_category(root_histograms, vars_dict, "SIGNAL", weight)
                fill_special_signal_histograms(
                    special_histograms_super[tree_key], vars_dict, weight)
            else:
                fill_category(root_histograms, vars_dict, "BG", weight)
                gid = vars_dict["genTauID"]
                if gid == -13:
                    fill_category(root_histograms, vars_dict, "BGMuon", weight)
                elif gid == -11:
                    fill_category(root_histograms, vars_dict, "BGEle", weight)
                elif gid == 0:
                    fill_category(root_histograms, vars_dict, "BGPion", weight)
                elif gid == 1:
                    fill_category(root_histograms, vars_dict, "BGRho", weight)
                elif gid == 10:
                    fill_category(root_histograms, vars_dict, "BGA1", weight)
                else:
                    other_BG_id[gid] = other_BG_id.get(gid, 0) + 1
                    fill_category(root_histograms, vars_dict, "BGOther", weight)

    return {
        "totalEvents":   totalEvents,
        "selectedEvents": selectedEvents,
        "sumWeights":    sumWeights,
        "sumWeightsP1":  sumWeightsP1,
        "sumWeightsM1":  sumWeightsM1,
    }


# ── Worker ────────────────────────────────────────────────────────────────────

def process_chunk_stage2(input_root, tree_keys, entry_range, config_bundle, worker_id):
    """
    Worker: abre input_root en READ, procesa entradas entry_range en todos los
    árboles de tree_keys, escribe histogramas parciales en un ROOT temporal.

    Devuelve (partial_file_path, counters_dict, other_BG_id_dict).
    """
    outputpath = config_bundle["outputpath"]

    # Logging propio del worker (strip inherited handlers)
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

    hist_config_v2   = config_bundle["hist_config_v2"]
    special_config   = config_bundle["special_config"]
    selectGEN        = config_bundle["selectGEN"]
    weight           = config_bundle["weight"]
    cuts_cfg         = config_bundle["cuts_cfg"]
    proccesing_cfg   = config_bundle["proccesing_cfg"]
    fileOutName_base = config_bundle["fileOutName_base"]

    # Abrir ROOT de entrada en modo READ — seguro para acceso concurrente
    infile = TFile.Open(input_root, "READ")
    if not infile or infile.IsZombie():
        raise RuntimeError(f"Worker {worker_id}: no se puede abrir {input_root}")

    # Obtener árboles
    trees = {}
    for key in tree_keys:
        t = infile.Get(f"outtree_{key}")
        if isinstance(t, ROOT.TTree):
            trees[key] = t
        else:
            logger.warning("Worker %d: árbol 'outtree_%s' no encontrado", worker_id, key)

    # Abrir pfile ANTES de crear histogramas: así ROOT los registra en pfile y
    # no pueden ser reclamados por el GC cuando infile.Close() se ejecute.
    partial_path = os.path.join(outputpath, f"partial_histos_worker{worker_id}_{fileOutName_base}.root")
    pfile = TFile(partial_path, "RECREATE")
    pfile.cd()

    # Construir histogramas dentro del worker (después del fork — seguro)
    base_histograms         = myutils.build_histogram_registry(hist_config_v2)
    base_special_histograms = myutils.set_up_root_histograms(special_config)

    root_histograms_super    = {"original": base_histograms}
    special_histograms_super = {"original": base_special_histograms}

    if "min_err" in trees:
        root_histograms_super["min_err"]    = myutils.clone_histograms_with_suffix(base_histograms, "_min")
        _reattach_histograms(root_histograms_super["min_err"], pfile)
        special_histograms_super["min_err"] = myutils.clone_histograms_with_suffix(base_special_histograms, "_min")
        _reattach_histograms(special_histograms_super["min_err"], pfile)
    if "max_err" in trees:
        root_histograms_super["max_err"]    = myutils.clone_histograms_with_suffix(base_histograms, "_max")
        _reattach_histograms(root_histograms_super["max_err"], pfile)
        special_histograms_super["max_err"] = myutils.clone_histograms_with_suffix(base_special_histograms, "_max")
        _reattach_histograms(special_histograms_super["max_err"], pfile)

    other_BG_id = {}

    start_entry, end_entry = entry_range
    counters = process_tree_range(
        rho_vars_extremes_trees=trees,
        root_histograms_super=root_histograms_super,
        special_histograms_super=special_histograms_super,
        weight=weight,
        selectGEN=selectGEN,
        cuts_cfg=cuts_cfg,
        proccesing_cfg=proccesing_cfg,
        logger_process=logger,
        other_BG_id=other_BG_id,
        start_entry=start_entry,
        end_entry=end_entry,
    )

    infile.Close()

    # Escribir histogramas parciales
    pfile.cd()
    for tree_key in root_histograms_super:
        write_histograms_recursive(root_histograms_super[tree_key])
    for tree_key in special_histograms_super:
        write_histograms_recursive(special_histograms_super[tree_key])
    pfile.Close()

    logger.info("Worker %d: terminado. Events=%d Selected=%d",
                worker_id, counters["totalEvents"], counters["selectedEvents"])
    return partial_path, counters, other_BG_id


# ── Merge histogramas ─────────────────────────────────────────────────────────

def merge_histogram_dicts_from_files(partial_files, root_histograms_super,
                                     special_histograms_super):
    """
    Para cada fichero parcial abre, obtiene cada histograma por nombre y
    llama h_main.Add(h_partial). Modifica in-place los dicts de histogramas.
    """
    # Construir mapa plano {nombre → objeto histograma en main}
    all_main = {}
    for hsuper in (root_histograms_super, special_histograms_super):
        for subtree in hsuper.values():
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
        help="Input ROOT file containing outtree_original "
             "(and optionally outtree_min_err, outtree_max_err)")
    parser.add_argument("--ang", type=float, default=[0.0, np.inf], nargs="+",
        help="Angular separation between decays (default: 0.0 to infinity)")
    parser.add_argument("--meson-cut", type=float, default=[0.0, np.inf], nargs="+",
        help="Meson cut range (default: 0.0 to infinity)")
    parser.add_argument("--lepton-cut", type=float, default=[0.0, np.inf], nargs="+",
        help="Lepton cut range (default: 0.0 to infinity)")
    parser.add_argument("--zmass-cut", type=float, default=[0.0, np.inf], nargs="+",
        help="Z mass cut range (default: 0.0 to infinity)")
    parser.add_argument("--compute-weights", action="store_true",
        help="Compute weights for polarization variations")
    parser.add_argument("--sin-eff", type=float, default=None,
        help="Effective sin^2 theta_W to use in weight calculations")
    parser.add_argument("--hist-config-v2", type=str,
        default="config/histograms/rho_analysis_config_v2.yml",
        help="Path to the v2 histogram config YAML (compact format)")
    parser.add_argument("--n-workers", type=int, default=None,
        help="Número de procesos paralelos (default: min(n_entries, n_cpus))")
    parser.add_argument("--cut", type=str, nargs="+", default=[], metavar="EXPR",
        help="Expresiones Python extra sobre vars_dict aplicadas en AND lógico. "
             "Ejemplo: --cut \"recoTauID == 1\" \"ZRecoMass > 85\"")


def main():
    # 1. Configuración
    general_configs = myutils.setup_analysis_config(
        _DEFAULT_CONFIG, _OUTPUT_BASE, parser_hook=my_hook)
    loggers    = general_configs["loggers"]
    run_config = general_configs["config"]
    args       = general_configs["args"]
    logger_config  = loggers["config"]
    logger_io      = loggers["io"]
    logger_process = loggers["processing"]

    selectDecay = general_configs["decay"]
    tauPCut     = run_config["cuts"]["tauCut"]

    angle_sep  = args.ang;   angle_sep  = [angle_sep[0], np.inf]  if len(angle_sep) == 1 else angle_sep
    meson_cut  = args.meson_cut;  meson_cut  = [meson_cut[0], np.inf]  if len(meson_cut) == 1 else meson_cut
    lepton_cut = args.lepton_cut; lepton_cut = [lepton_cut[0], np.inf] if len(lepton_cut) == 1 else lepton_cut
    zmass_cut  = args.zmass_cut;  zmass_cut  = [zmass_cut[0], np.inf]  if len(zmass_cut) == 1 else zmass_cut

    input_root = args.tree_file
    if not os.path.isfile(input_root):
        logger_io.error("Input ROOT file %s not found", input_root)
        sys.exit(1)

    # 2. Cargar config de histogramas
    with open(args.hist_config_v2, "r") as f:
        hist_config_v2 = yaml.safe_load(f)
    special_config = hist_config_v2.get("special", {})

    # 3. Detectar árboles disponibles y número de entradas
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
    infile.Close()  # CERRAR antes del fork

    selectGEN = selectDecay
    if selectDecay == 2:
        selectGEN = 1
    weight = 1.0
    cuts_cfg = {
        "tauPCut": tauPCut, "meson_cut": meson_cut,
        "lepton_cut": lepton_cut, "zmass_cut": zmass_cut, "angle_sep": angle_sep,
        "extra_cuts": args.cut,
    }
    proccesing_cfg = {
        "sin_eff": args.sin_eff,
        "compute_weights": args.compute_weights,
        "decay_mode": selectDecay,
    }

    outputpath = os.path.dirname(input_root)
    out_histos_string = "Histos_"
    if angle_sep[0] > 0:
        out_histos_string += f"dRgt{angle_sep[0]}_{angle_sep[1]}_"
    if meson_cut[0] > 0 or meson_cut[1] < 100:
        out_histos_string += f"MesonPgt{meson_cut[0]}_lt{meson_cut[1]}_"
    if lepton_cut[0] > 0 or lepton_cut[1] < 100:
        out_histos_string += f"LeptonPgt{lepton_cut[0]}_lt{lepton_cut[1]}_"
    if zmass_cut[0] > 0 or zmass_cut[1] < 200:
        out_histos_string += f"Zmassgt{zmass_cut[0]}_lt{zmass_cut[1]}_"
    if args.cut:
        safe = "_".join(e.replace(" ", "").replace("==", "eq").replace(">", "gt").replace("<", "lt") for e in args.cut)
        out_histos_string += f"cut_{safe}_"
    fileOutName = os.path.join(outputpath, out_histos_string + general_configs["fileOutName"])
    fileOutName_base = Path(fileOutName).stem
    os.makedirs(outputpath, exist_ok=True)

    # 4. Determinar número de workers
    n_workers = args.n_workers or min(n_entries, os.cpu_count() or 1)
    logger_io.info("n_entries=%d, n_workers=%d", n_entries, n_workers)

    # 5. Modo secuencial
    if n_workers == 1:
        logger_io.info("Sequential mode.")
        from RhoAnalysis.temp_functions import process_tree

        base_histograms         = myutils.build_histogram_registry(hist_config_v2)
        base_special_histograms = myutils.set_up_root_histograms(special_config)
        infile = TFile.Open(input_root, "READ")
        trees = {}
        for key in tree_keys:
            t = infile.Get(f"outtree_{key}")
            if isinstance(t, ROOT.TTree):
                trees[key] = t

        root_histograms_super    = {"original": base_histograms}
        special_histograms_super = {"original": base_special_histograms}
        if "min_err" in trees:
            root_histograms_super["min_err"]    = myutils.clone_histograms_with_suffix(base_histograms, "_min")
            special_histograms_super["min_err"] = myutils.clone_histograms_with_suffix(base_special_histograms, "_min")
        if "max_err" in trees:
            root_histograms_super["max_err"]    = myutils.clone_histograms_with_suffix(base_histograms, "_max")
            special_histograms_super["max_err"] = myutils.clone_histograms_with_suffix(base_special_histograms, "_max")

        other_BG_id = {}
        counters = process_tree(
            rho_vars_extremes_trees=trees,
            root_histograms_super=root_histograms_super,
            special_histograms_super=special_histograms_super,
            weight=weight, selectGEN=selectGEN,
            cuts_cfg=cuts_cfg, proccesing_cfg=proccesing_cfg,
            logger_process=logger_process, other_BG_id=other_BG_id,
        )
        infile.Close()
        _write_output(fileOutName, root_histograms_super, special_histograms_super,
                      counters, other_BG_id, logger_io)
        return

    # 6. Modo paralelo
    entry_ranges = split_entry_ranges(n_entries, n_workers)
    n_chunks     = len(entry_ranges)

    config_bundle = {
        "hist_config_v2":   hist_config_v2,
        "special_config":   special_config,
        "selectGEN":        selectGEN,
        "weight":           weight,
        "cuts_cfg":         cuts_cfg,
        "proccesing_cfg":   proccesing_cfg,
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
                process_chunk_stage2,
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

    # 7. Reconstruir histogramas vacíos en main y fusionar parciales
    base_histograms         = myutils.build_histogram_registry(hist_config_v2)
    base_special_histograms = myutils.set_up_root_histograms(special_config)

    root_histograms_super    = {"original": base_histograms}
    special_histograms_super = {"original": base_special_histograms}
    if "min_err" in tree_keys:
        root_histograms_super["min_err"]    = myutils.clone_histograms_with_suffix(base_histograms, "_min")
        special_histograms_super["min_err"] = myutils.clone_histograms_with_suffix(base_special_histograms, "_min")
    if "max_err" in tree_keys:
        root_histograms_super["max_err"]    = myutils.clone_histograms_with_suffix(base_histograms, "_max")
        special_histograms_super["max_err"] = myutils.clone_histograms_with_suffix(base_special_histograms, "_max")

    logger_io.info("Merging %d partial files...", len(partial_files))
    merge_histogram_dicts_from_files(partial_files, root_histograms_super, special_histograms_super)

    # Limpiar parciales
    for p in partial_files:
        try:
            os.remove(p)
        except OSError:
            pass

    # Sumar contadores y BG ids
    merged_counters = {
        "totalEvents":   sum(c["totalEvents"]   for c in all_counters),
        "selectedEvents": sum(c["selectedEvents"] for c in all_counters),
        "sumWeights":    sum(c["sumWeights"]    for c in all_counters),
        "sumWeightsP1":  sum(c["sumWeightsP1"]  for c in all_counters),
        "sumWeightsM1":  sum(c["sumWeightsM1"]  for c in all_counters),
    }
    merged_bg_ids = {}
    for d in all_bg_ids:
        for k, v in d.items():
            merged_bg_ids[k] = merged_bg_ids.get(k, 0) + v

    logger_io.info("Total: events=%d selected=%d (%.1fs)",
                   merged_counters["totalEvents"], merged_counters["selectedEvents"],
                   time.time() - t_start)

    _write_output(fileOutName, root_histograms_super, special_histograms_super,
                  merged_counters, merged_bg_ids, logger_io)


def _write_output(fileOutName, root_histograms_super, special_histograms_super,
                  counters, other_BG_id, logger_io):
    """Escribe el ROOT de salida con histogramas y el CSV de BG ids."""
    import pandas as pd
    logger_io.info("Writing output ROOT file %s", fileOutName)
    logger_io.info("Events=%d Selected=%d", counters["totalEvents"], counters["selectedEvents"])

    outfile = TFile(fileOutName, "RECREATE")
    outfile.cd()
    for tree_key in root_histograms_super:
        write_histograms_recursive(root_histograms_super[tree_key])
    for tree_key in special_histograms_super:
        write_histograms_recursive(special_histograms_super[tree_key])

    # Proyecciones de Omega_GEN_ZGenMass
    hist2d      = root_histograms_super["original"]["Gen"]["Omega_GEN_ZGenMass"]["SIGNAL"]["nominal"]
    hist2d_P1   = root_histograms_super["original"]["Gen"]["Omega_GEN_ZGenMass"]["SIGNAL"]["P1"]
    hist2d_M1   = root_histograms_super["original"]["Gen"]["Omega_GEN_ZGenMass"]["SIGNAL"]["M1"]
    nbins       = hist2d.GetNbinsY()
    bin_edges   = [hist2d.GetYaxis().GetBinLowEdge(i) for i in range(1, nbins + 2)]
    last_bin    = 0
    for i in range(1, nbins + 1):
        if hist2d.ProjectionX(f"_tmp_{i}", i, i).GetEntries() > 0:
            last_bin = i
    n = 10
    for i in range(last_bin - n, last_bin + 1, 2):
        proj_name = f"Omega_GEN_SIGNAL_ZGenMass_ProjBin_{round(bin_edges[i], 3)}"
        hist2d.ProjectionX(proj_name, i, i+1).Write()
    for i in range(last_bin - n, last_bin + 1, 2):
        proj_name = f"Omega_GEN_SIGNAL_ZGenMass_ProjBin_{round(bin_edges[i], 3)}_M1"
        hist2d_M1.ProjectionX(proj_name, i, i+1).Write()
    for i in range(last_bin - n, last_bin + 1, 2):
        proj_name = f"Omega_GEN_SIGNAL_ZGenMass_ProjBin_{round(bin_edges[i], 3)}_P1"
        hist2d_P1.ProjectionX(proj_name, i, i+1).Write()
    
    # Crear histograma sumando todos los bines menos el último
    first_proj = hist2d.ProjectionX("temp_first", last_bin - n, last_bin - 1)
    h_all_but_last = first_proj.Clone("Omega_GEN_SIGNAL_ZGenMass_AllBinsExceptLast")

    for i in range(last_bin - n + 2, last_bin - 1, 2):
        proj = hist2d.ProjectionX("temp_proj", i, i + 1)
        h_all_but_last.Add(proj)

    h_all_but_last.Write()

    # Lo mismo para M1
    first_proj_M1 = hist2d_M1.ProjectionX("temp_first_M1", last_bin - n, last_bin - 1)
    h_all_but_last_M1 = first_proj_M1.Clone("Omega_GEN_SIGNAL_ZGenMass_AllBinsExceptLast_M1")

    for i in range(last_bin - n + 2, last_bin - 1, 2):
        proj = hist2d_M1.ProjectionX("temp_proj_M1", i, i + 1)
        h_all_but_last_M1.Add(proj)

    h_all_but_last_M1.Write()

    # Lo mismo para P1
    first_proj_P1 = hist2d_P1.ProjectionX("temp_first_P1", last_bin - n, last_bin - 1)
    h_all_but_last_P1 = first_proj_P1.Clone("Omega_GEN_SIGNAL_ZGenMass_AllBinsExceptLast_P1")

    for i in range(last_bin - n + 2, last_bin - 1, 2):
        proj = hist2d_P1.ProjectionX("temp_proj_P1", i, i + 1)
        h_all_but_last_P1.Add(proj)

    h_all_but_last_P1.Write()

    outfile.Close()

    output_name = fileOutName.replace(".root", "_otherBGid.csv")
    df_other_BG = pd.DataFrame(sorted(other_BG_id.items()), columns=["genTauID", "count"])
    df_other_BG.to_csv(output_name, index=False)
    logger_io.info("Other BG IDs saved to %s", output_name)
    logger_io.info("Done. Results in %s", fileOutName)


if __name__ == "__main__":
    main()
