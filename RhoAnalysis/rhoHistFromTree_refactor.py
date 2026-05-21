#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Re-histogramming script for tau-rho analysis.

Lee un ROOT que contiene:
  - outtree_original
  - opcionalmente outtree_min_err, outtree_max_err

y genera un nuevo ROOT con los mismos histogramas que el análisis
original, usando el mismo sistema de configuración (YAML + myutils).

Uso típico:

  python rehist_from_trees.py \
      --config config/default/taurecolong.yaml \
      --tree-file path/al/fichero_con_trees.root

El nombre de salida se toma de general_configs["fileOutName"]
y se escribe en general_configs["outputpath"].
"""
import numpy as np
import os
import sys
import math
import pprint
import yaml
import copy
import ROOT
# import matplotlib.pyplot as plt
from ROOT import TFile
# No necesitamos TTree, TH1F, TH2F explícitamente, solo ROOT.TTree & co.

from modules import myutils  # mismo sistema de config que el código original
from modules import pi0Reco
from modules.plotting import plot_sigma_vs_energy_root
from modules import optimalVariabRho
from RhoAnalysis.temp_functions import process_tree, fill_special_signal_histograms

# ---------------------------------------------------------------------
# Función auxiliar: escribir recursivamente todos los histogramas ROOT
# contenidos en diccionarios anidados.
# ---------------------------------------------------------------------
def write_histograms_recursive(obj):
    """
    Recorre un diccionario anidado y ejecuta `.Write()` en cada histograma ROOT.
    """
    if isinstance(obj, dict):
        for value in obj.values():
            write_histograms_recursive(value)
    else:
        try:
            obj.Write()
        except AttributeError:
            print(f"Objeto {obj} no tiene método .Write(). Ignorado.")


# ---------------------------------------------------------------------
# Configuración por defecto (igual que en el análisis original)
# ---------------------------------------------------------------------
default_config = "config/default/taurecolong.yaml"
outputbasepath = "Results/RhoAnalysis/"


def my_hook(parser):
    """
    Hook para añadir argumentos específicos de este script, manteniendo
    el mismo sistema de configuración que el código original.
    """

    # Fichero ROOT de entrada con los árboles outtree_*
    parser.add_argument(
        "--tree-file",
        type=str,
        required=True,
        help="Input ROOT file containing outtree_original "
             "(and optionally outtree_min_err, outtree_max_err)",
    )
    parser.add_argument(
        "--ang",
        type=float,
        default=[0.0, np.inf],
        nargs="+",
        help="Angular separation between decays (default: 0.0 to infinity)",
    )
    parser.add_argument(
        "--meson-cut",
        type=float,
        default=[0.0, np.inf],
        nargs="+",
        help="Meson cut range (default: 0.0 to infinity)",
    )
    parser.add_argument(
        "--lepton-cut",
        type=float,
        default=[0.0, np.inf],
        nargs="+",
        help="Lepton cut range (default: 0.0 to infinity)",
    )
    parser.add_argument(
        "--zmass-cut",
        type=float,
        default=[0.0, np.inf],
        nargs="+",
        help="Z mass cut range (default: 0.0 to infinity)",
    )
    parser.add_argument("--compute-weights", action="store_true",
        help="Compute weights for polarization variations"
    )
    parser.add_argument("--sin-eff", type=float, default=None,
        help="Effective sin^2 theta_W to use in weight calculations"
    )
    parser.add_argument(
        "--hist-config-v2",
        type=str,
        default="config/histograms/rho_analysis_config_v2.yml",
        help="Path to the v2 histogram config YAML (compact format)",
    )



def main():
    # -----------------------------------------------------------------
    # Inicializar configuración y loggers con el mismo sistema que antes
    # -----------------------------------------------------------------
    general_configs = myutils.setup_analysis_config(
        default_config,
        outputbasepath,
        parser_hook=my_hook,
    )

    loggers = general_configs["loggers"]
    run_config = general_configs["config"]
    args = general_configs["args"]

    logger_config = loggers["config"]
    logger_io = loggers["io"]
    logger_process = loggers["processing"]

    logger_config.info("Configuration loaded!")
    logger_config.info(
        "Configuration:\n%s",
        pprint.pformat(general_configs, indent=4),
    )

    # Parámetros relevantes
    sample = run_config["general"]["sample"]
    selectDecay = general_configs["decay"]
    tauPCut = run_config["cuts"]["tauCut"]
    angle_sep = args.ang
    if len(angle_sep) == 1:
        angle_sep = [angle_sep[0], np.inf]
    meson_cut = args.meson_cut
    if len(meson_cut) == 1:
        meson_cut = [meson_cut[0], np.inf]
    lepton_cut = args.lepton_cut
    if len(lepton_cut) == 1:
        lepton_cut = [lepton_cut[0], np.inf]
    zmass_cut = args.zmass_cut
    if len(zmass_cut) == 1:
        zmass_cut = [zmass_cut[0], np.inf]
    # Ruta del ROOT de entrada con los árboles
    input_root = args.tree_file
    input_root_path = os.path.dirname(input_root)
    
    
    if not os.path.isfile(input_root):
        logger_io.error("Input ROOT file %s not found", input_root)
        sys.exit(1)

    logger_io.info("Using input tree file: %s", input_root)

    
    # -----------------------------------------------------------------
    # Configuración de histogramas (formato v2 compacto)
    # -----------------------------------------------------------------
    hist_config_v2_path = args.hist_config_v2
    with open(hist_config_v2_path, "r") as f:
        hist_config_v2 = yaml.safe_load(f)
    logger_config.info("Loaded v2 histogram config from %s", hist_config_v2_path)

    # Histogramas estándar: dict 4-niveles hists[level][variable][category][weight]
    base_histograms = myutils.build_histogram_registry(hist_config_v2)

    # Histogramas especiales (Pi0Mass, ZVisMass bins): dict v1 para fill explícito
    special_config = hist_config_v2.get("special", {})
    base_special_histograms = myutils.set_up_root_histograms(special_config)

    # Abrir fichero ROOT de entrada
    infile = TFile.Open(input_root, "READ")
    if not infile or infile.IsZombie():
        logger_io.error("Could not open input ROOT file %s", input_root)
        sys.exit(1)

    theta_bins = {
        "barrel":     [50, 130],
        "endcap":     [15, 40],
        "transition": [40, 50]
    }    
    # Convertir grados a radianes
    theta_bins_rad = {
        region: (math.radians(a), math.radians(b))
        for region, (a, b) in theta_bins.items()
    }

    # ---- Definir bins de energía ----
    energy_bins = [
        [0, 2.5],
        [2.5, 7.5],
        [7.5, 12.5],
        [12.5, np.inf]
    ]

    # ---- Crear histogramas vacíos ----
    bin_edges = np.linspace(0, 1, 51)  # 50 bines en el rango 0–1
    n_bins = len(bin_edges) - 1

    # Diccionario con histogramas inicializados a cero
    hist_dict = {
        region: {i: [] for i in range(len(energy_bins))}
        for region in theta_bins_rad.keys()
    }
    tree_name = "outtree"

    # Detectar qué árboles están presentes en el fichero
    trees = {}
    for key in ["original", "min_err", "max_err"]:
        full_name = f"{tree_name}_{key}"
        tree = infile.Get(full_name)
        if isinstance(tree, ROOT.TTree) and tree.GetEntries() > 0:
            trees[key] = tree
            logger_io.info(
                "Found tree '%s' with %d entries",
                full_name,
                tree.GetEntries(),
            )
    if "original" not in trees:
        logger_io.error(
            "Tree 'outtree_original' not found or empty in %s", input_root
        )
        sys.exit(1)

    # -----------------------------------------------------------------
    # Crear el "super" diccionario de histogramas
    # -----------------------------------------------------------------
    root_histograms_super = {"original": base_histograms}
    special_histograms_super = {"original": base_special_histograms}
    resolution_histograms = {"original": hist_dict}

    # Si existen los árboles de extremos, clonamos con sufijos _min y _max.
    # clone_histograms_with_suffix es recursivo: funciona con la estructura v2.
    if "min_err" in trees:
        logger_io.info("Cloning histograms for min_err with suffix '_min'")
        root_histograms_super["min_err"] = myutils.clone_histograms_with_suffix(
            base_histograms, "_min"
        )
        special_histograms_super["min_err"] = myutils.clone_histograms_with_suffix(
            base_special_histograms, "_min"
        )
        resolution_histograms["min_err"] = copy.deepcopy(hist_dict)

    if "max_err" in trees:
        logger_io.info("Cloning histograms for max_err with suffix '_max'")
        root_histograms_super["max_err"] = myutils.clone_histograms_with_suffix(
            base_histograms, "_max"
        )
        special_histograms_super["max_err"] = myutils.clone_histograms_with_suffix(
            base_special_histograms, "_max"
        )
        resolution_histograms["max_err"] = copy.deepcopy(hist_dict)

    # -----------------------------------------------------------------
    # Rehacer histogramas a partir de los árboles
    # -----------------------------------------------------------------
    weight = 1.0
    totalEvents = 0
    selectedEvents = 0
    sumWeights = 0.0
    sumWeightsP1 = 0.0
    sumWeightsM1 = 0.0

    # Selección de genTauID que se considera "signal", igual que en el análisis
    selectGEN = selectDecay
    if selectDecay == 2:
        selectGEN = 1
    
    other_BG_id = dict()
    cuts_cfg = {
        "tauPCut": tauPCut,
        "meson_cut": meson_cut,
        "lepton_cut": lepton_cut,
        "zmass_cut": zmass_cut,
        "angle_sep": angle_sep
        }
    proccesing_cfg = {
        "sin_eff" : args.sin_eff,
        "compute_weights": args.compute_weights
    }
    counters = process_tree(
                rho_vars_extremes_trees=trees,
                root_histograms_super=root_histograms_super,
                special_histograms_super=special_histograms_super,
                weight=weight,
                selectGEN=selectGEN,
                cuts_cfg=cuts_cfg,
                proccesing_cfg=proccesing_cfg,
                logger_process=logger_process,
                other_BG_id=other_BG_id,
    )
    totalEvents   = counters["totalEvents"]
    selectedEvents = counters["selectedEvents"]
    sumWeights    = counters["sumWeights"]
    sumWeightsP1  = counters["sumWeightsP1"]
    sumWeightsM1  = counters["sumWeightsM1"]


    infile.Close()

    # -----------------------------------------------------------------
    # Guardar config y escribir ROOT de salida con histogramas
    # -----------------------------------------------------------------
    outputpath = input_root_path
    fileOutName = os.path.join(outputpath, "Histos_" + general_configs["fileOutName"])
    out_histos_string = "Histos_"
    if angle_sep[0] > 0:
        out_histos_string += f"dRgt{angle_sep[0]}_{angle_sep[1]}_"
    if meson_cut[0] > 0 or meson_cut[1] < 100:
        out_histos_string += f"MesonPgt{meson_cut[0]}_lt{meson_cut[1]}_"
    if lepton_cut[0] > 0 or lepton_cut[1] < 100:
        out_histos_string += f"LeptonPgt{lepton_cut[0]}_lt{lepton_cut[1]}_"
    if zmass_cut[0] > 0 or zmass_cut[1] < 200:
        out_histos_string += f"Zmassgt{zmass_cut[0]}_lt{zmass_cut[1]}_"
    
    fileOutName = os.path.join(outputpath, out_histos_string + general_configs["fileOutName"])
    
    os.makedirs(outputpath, exist_ok=True)

    logger_io.info(
        "Run over %d events (original tree), selected %d",
        totalEvents,
        selectedEvents,
    )
    logger_io.info(
        "Weights? sumW = %f, sumW_P1 = %f, sumW_M1 = %f",
        sumWeights,
        sumWeightsP1,
        sumWeightsM1,
    )
    logger_io.info("Writing output ROOT file %s", fileOutName)

      
    outfile = TFile(fileOutName, "RECREATE")
    outfile.cd()
    
    for tree_key in root_histograms_super:
        write_histograms_recursive(root_histograms_super[tree_key])
    for tree_key in special_histograms_super:
        write_histograms_recursive(special_histograms_super[tree_key])

    # Generate projection bins of Omega_GEN_ZGenMass (v2 dict structure)
    hist2d_hist    = root_histograms_super["original"]["Gen"]["Omega_GEN_ZGenMass"]["SIGNAL"]["nominal"]
    hist2d_hist_P1 = root_histograms_super["original"]["Gen"]["Omega_GEN_ZGenMass"]["SIGNAL"]["P1"]
    hist2d_hist_M1 = root_histograms_super["original"]["Gen"]["Omega_GEN_ZGenMass"]["SIGNAL"]["M1"]
    nbins = hist2d_hist.GetNbinsY()
    # get bin edges
    bin_edges = [hist2d_hist.GetYaxis().GetBinLowEdge(i) for i in range(1, nbins+2)]
    
    # Get last bin with content
    last_bin_with_content = 0
    for i in range(1, nbins+1):
        if hist2d_hist.ProjectionX(f"Omega_GEN_SIGNAL_ZGenMass_ProjBin_{bin_edges[i]}", i, i).GetEntries() > 0:
            last_bin_with_content = i
    logger_io.info(f"Last bin with content in Omega_GEN_SIGNAL_ZGenMass: {bin_edges[last_bin_with_content]} GeV")
    
    # Get the las n bins
    n = 10
    for i in range(last_bin_with_content-n, last_bin_with_content+1):
        proj_name = f"Omega_GEN_SIGNAL_ZGenMass_ProjBin_{round(bin_edges[i],2)}"
        TH1D_proj = hist2d_hist.ProjectionX(proj_name, i, i)
        TH1D_proj.Write()

    for i in range(last_bin_with_content-n, last_bin_with_content+1):
        proj_name = f"Omega_GEN_SIGNAL_ZGenMass_ProjBin_{round(bin_edges[i],2)}_M1"
        TH1D_proj = hist2d_hist_M1.ProjectionX(proj_name, i, i)
        TH1D_proj.Write()

    for i in range(last_bin_with_content-n, last_bin_with_content+1):
        proj_name = f"Omega_GEN_SIGNAL_ZGenMass_ProjBin_{round(bin_edges[i],2)}_P1"
        TH1D_proj = hist2d_hist_P1.ProjectionX(proj_name, i, i)
        TH1D_proj.Write()
    # for res_hist_key in resolution_histograms:
    #     # for region in resolution_histograms[res_hist_key]:
    #     #     print(resolution_histograms[res_hist_key][region])
    #     canvas, multigraph, sigma_results = plot_sigma_vs_energy_root(resolution_histograms[res_hist_key], bin_edges=bin_edges, energy_bins=energy_bins)
    #     if canvas is not None:
    #         sigma_energy_fit = myutils.fit_sigma_energy(sigma_results)
    #     for graph in multigraph:
    #         graph[0].Write(f"Graph_PhotonPRes_vs_Energy_{res_hist_key}_{graph[1]}")
    #     # multigraph.Write(f"PhotonPRes_vs_Energy_{res_hist_key}")
    #     if canvas is not None:
    #         canvas.Write(f"Canvas_PhotonPRes_vs_Energy_{res_hist_key}")
    #         # output_png = os.path.join(outputpath, f"PhotonPRes_vs_Energy_{res_hist_key}_d{selectDecay}.png")
    #         # canvas.SaveAs(output_png)
    outfile.Close()
    other_BG_id_list = sorted(other_BG_id.items())
    # Save as csv
    output_name = fileOutName.replace(".root", "_otherBGid.csv")
    import pandas as pd
    df_other_BG = pd.DataFrame(other_BG_id_list, columns=["genTauID", "count"])
    df_other_BG.to_csv(output_name, index=False)
    logger_io.info(f"Other BG IDs saved to {output_name}")
    logger_io.info(f"All histograms written and file closed. Results in {fileOutName}")


if __name__ == "__main__":
    main()
