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
    # Configuración de histogramas (misma que el análisis original)
    # -----------------------------------------------------------------
    histogram_config = general_configs.get("histograms_config", {})
    base_histograms = myutils.set_up_root_histograms(histogram_config)

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
    # Crear el "super" diccionario de histogramas, igual que en el análisis
    # -----------------------------------------------------------------
    root_histograms_super = {"original": base_histograms}
    # extra_plt_plots = {"original": None}
    resolution_histograms = {"original": hist_dict}

    # Si existen los árboles de extremos, clonamos con sufijos _min y _max
    if "min_err" in trees:
        logger_io.info("Cloning histograms for min_err with suffix '_min'")
        root_histograms_super["min_err"] = myutils.clone_histograms_with_suffix(
            base_histograms, '_min'
        )
        # extra_plt_plots["min_err"] = None
        resolution_histograms["min_err"] = copy.deepcopy(hist_dict)

    if "max_err" in trees:
        logger_io.info("Cloning histograms for max_err with suffix '_max'")
        root_histograms_super["max_err"] = myutils.clone_histograms_with_suffix(
            base_histograms, '_max'
        )
        # extra_plt_plots["max_err"] = None
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
    

    for tree_key, tree in trees.items():
        root_histograms = root_histograms_super[tree_key]
        logger_process.info(
            "Refilling histograms for tree key '%s' with %d entries",
            tree_key,
            tree.GetEntries(),
        )
        hist_dict = resolution_histograms[tree_key]
        n_entries = tree.GetEntries()
        for i in range(n_entries):
            tree.GetEntry(i)
            entry = tree  # para que sea más corto escribir

            # Contadores globales solo para el árbol "original"
            if tree_key == "original":
                totalEvents += 1

            # Extraer variables del árbol (mismas ramas que definiste)
            try:
                beamE = float(entry.beamE)
                recoMesonE = float(entry.recoMesonE)
                recoMesonTheta = float(entry.recoMesonTheta)
                recoMesonPhi = float(entry.recoMesonPhi)
                recoMesonP = float(entry.recoMesonP)

                genMesonE = float(entry.genMesonE)
                genMesonTheta = float(entry.genMesonTheta)

                cos_theta = float(entry.cos_theta)
                cos_psi = float(entry.cos_psi)
                cos_theta_rho = float(entry.cos_theta_rho)

                gen_w = float(entry.genOmega)
                w = float(entry.omega)

                gen_cos_theta = float(entry.gen_cos_theta)
                gen_cos_psi = float(entry.gen_cos_psi)
                gen_cos_theta_tau = float(entry.gen_cos_theta_tau)

                genTauID = int(entry.genTauID)
                recoTauID = int(entry.recoTauID)

                # Pesos almacenados en el árbol
                weight_P1 = float(entry.weight_P1)
                weight_M1 = float(entry.weight_M1)
                
                leptonP = float(entry.lepP)
                leptonE = float(entry.lepE)
                leptonPhi = float(entry.lepPhi)
                leptonTheta = float(entry.lepTheta)
                
                mesonp4 = ROOT.TLorentzVector()
                mesonp4.SetPxPyPzE(
                    recoMesonP * math.sin(recoMesonTheta) * math.cos(recoMesonPhi),
                    recoMesonP * math.sin(recoMesonTheta) * math.sin(recoMesonPhi),
                    recoMesonP * math.cos(recoMesonTheta),
                    recoMesonE
                )
                leptonp4 = ROOT.TLorentzVector()
                leptonp4.SetPxPyPzE(
                    leptonP * math.sin(leptonTheta) * math.cos(leptonPhi),
                    leptonP * math.sin(leptonTheta) * math.sin(leptonPhi),
                    leptonP * math.cos(leptonTheta), leptonE
                )

            except AttributeError as e:
                logger_process.error(
                    "Missing branch in tree '%s': %s", tree_key, e
                )
                continue

            # Variable x definida en el análisis original
            if beamE != 0:
                x = 2.0 * recoMesonE / beamE - 1.0
            else:
                x = 0.0
            if recoMesonP < tauPCut:
                continue
            
            z_p4 = mesonp4 + leptonp4 
            zmass = z_p4.M()
            if recoMesonP < meson_cut[0] or recoMesonP > meson_cut[1]:
                continue # Ignoring this event to get the bk
            if leptonP < lepton_cut[0] or leptonP > lepton_cut[1]:
                continue # Ignoring this event to get the bk
            if zmass < zmass_cut[0] or zmass > zmass_cut[1]:
                continue # Ignoring this event to get the bk
            
            dR_between = myutils.dRAngle(mesonp4, leptonp4)
            if dR_between < angle_sep[0] or dR_between > angle_sep[1]:
                continue # Ignoring this event to get the bk
            if genTauID == selectGEN:
                root_histograms["Reco"]["Events"]["DeltaR_LepMeson_SIGNAL"].Fill(dR_between)
                root_histograms["Reco"]["Events"]["MesonP_SIGNAL"].Fill(recoMesonP)
                root_histograms["Reco"]["Events"]["LeptonP_SIGNAL"].Fill(leptonP)
            else:
                root_histograms["Reco"]["Events"]["DeltaR_LepMeson_BG"].Fill(dR_between)
                root_histograms["Reco"]["Events"]["MesonP_BG"].Fill(recoMesonP)
                root_histograms["Reco"]["Events"]["LeptonP_BG"].Fill(leptonP)
            
            
            # pt a partir de P y theta
            recoMesonPt = recoMesonP * math.sin(recoMesonTheta)

            # -----------------------------------------------------------------
            # Histogramas "ALL"
            # -----------------------------------------------------------------

            root_histograms["Reco"]["Events"]["RecoMesonEOverBeamE_ALL"].Fill(
                recoMesonE / beamE, weight
            )

            root_histograms["Reco"]["Events"]["RecoMesonCosTheta_ALL"].Fill(
                math.cos(recoMesonTheta), weight
            )
            root_histograms["Reco"]["Events"]["CosTheta_ALL"].Fill(
                cos_theta, weight
            )
            root_histograms["Reco"]["Events"]["CosPsi_ALL"].Fill(
                cos_psi, weight
            )

            root_histograms["Gen"]["Events"]["Omega_GEN_ALL"].Fill(
                gen_w, weight
            )
            root_histograms["Gen"]["Events"]["CosTheta_GEN_ALL"].Fill(
                gen_cos_theta, weight
            )
            root_histograms["Gen"]["Events"]["CosPsi_GEN_ALL"].Fill(
                gen_cos_psi, weight
            )

            root_histograms["Gen"]["Events"]["CosThetaTau_GEN_ALL"].Fill(
                gen_cos_theta_tau, weight
            )
            root_histograms["Gen"]["Events"]["CosThetaRho_GEN_ALL"].Fill(
                math.cos(genMesonTheta), weight
            )
            root_histograms["Gen"]["Events"]["CosThetaRho_ALL"].Fill(
                math.cos(recoMesonTheta), weight
            )

            root_histograms["Reco"]["Events"]["OmegaCosTheta_ALL"].Fill(
                w, cos_theta_rho, weight
            )
            root_histograms["Gen"]["Events"]["OmegaCosThetaTau_GEN_ALL"].Fill(
                gen_w, gen_cos_theta_tau, weight
            )

            root_histograms["Reco"]["Events"]["RecoMeson_X_ALL"].Fill(
                x, weight
            )

            # -----------------------------------------------------------------
            # Clasificación SIGNAL vs BG, igual que en el código original
            # (usando genTauID y selectGEN)
            # -----------------------------------------------------------------
            if genTauID == selectGEN:
                # Consideramos este evento como señal
                if tree_key == "original":
                    selectedEvents += 1
                    sumWeights += weight
                    sumWeightsP1 += weight * weight_P1
                    sumWeightsM1 += weight * weight_M1
                
                if genTauID == 1:
                    # Pi0 signal
                    reco_pi0_mass = pi0Reco.compute_pi0_mass(entry)
                    gen_pi0_mass_smear, smeared_p4 = pi0Reco.compute_gen_pi0_mass_with_smearing(entry)
                    gen_pi0_mass = pi0Reco.compute_pi0_mass(entry, reco=False)
                    recoP4 = pi0Reco.compute_photonp4(entry)
                    genP4 = pi0Reco.compute_photonp4(entry, reco=False)
                    if len(smeared_p4) == 2 and len(recoP4) == 2 and len(genP4) == 2:
                        # print("Calculando Resolución")
                        (res1_gen_smeared, res2_gen_smeared), (angle_1_smeared, angle_2_smeared), (E_gen_1_smeared, E_gen_2_smeared) = myutils.compute_photon_resolution_two_by_two(smeared_p4, genP4)
                        (res1_reco, res2_reco), (angle_1, angle_2), (E_gen_1, E_gen_2) = myutils.compute_photon_resolution_two_by_two(recoP4, genP4)
                        # print(f"Res reco: {res1_reco}, {res2_reco}")
                        # print(f"Res gen smeared: {res1_gen_smeared}, {res2_gen_smeared}")
                        root_histograms["Reco"]["Events"]["Pi0Mass_SIGNAL"].Fill(reco_pi0_mass)
                        root_histograms["Gen"]["Events"]["Pi0Mass_GEN_SIGNAL_smeared"].Fill(gen_pi0_mass_smear)
                        root_histograms["Gen"]["Events"]["Pi0Mass_GEN_SIGNAL"].Fill(gen_pi0_mass)
                        
                        root_histograms["Reco"]["Events"]["PhotonPRes"].Fill(res1_reco)
                        root_histograms["Reco"]["Events"]["PhotonPRes"].Fill(res2_reco)
                        
                        root_histograms["Gen"]["Events"]["PhotonPRes"].Fill(res1_gen_smeared)
                        root_histograms["Gen"]["Events"]["PhotonPRes"].Fill(res2_gen_smeared)
                        hist_dict = myutils.update_resolution_hist(res1_reco, E_gen_1, angle_1,
                           theta_bins_rad,
                           energy_bins,
                           hist_dict,
                           bin_edges)
                        hist_dict = myutils.update_resolution_hist(res2_reco, E_gen_2, angle_2,
                           theta_bins_rad,
                           energy_bins,
                           hist_dict,
                           bin_edges)
                
                # Matched (GEN) histos
                root_histograms["Matched"]["Events"]["MesonEOverBeamE"].Fill(
                    genMesonE / beamE, weight
                )
                root_histograms["Matched"]["Events"]["MesonEOverBeamE_P1"].Fill(
                    genMesonE / beamE, weight_P1 * weight
                )
                root_histograms["Matched"]["Events"]["MesonEOverBeamE_M1"].Fill(
                    genMesonE / beamE, weight_M1 * weight
                )

                # Reco omega
                root_histograms["Reco"]["Events"]["Omega_SIGNAL"].Fill(
                    w, weight
                )
                root_histograms["Reco"]["Events"]["Omega_SIGNAL_P1"].Fill(
                    w, weight * weight_P1
                )
                root_histograms["Reco"]["Events"]["Omega_SIGNAL_M1"].Fill(
                    w, weight * weight_M1
                )

                # Gen omega
                root_histograms["Gen"]["Events"]["Omega_GEN_SIGNAL"].Fill(
                    gen_w, weight
                )
                root_histograms["Gen"]["Events"]["Omega_GEN_SIGNAL_P1"].Fill(
                    gen_w, weight * weight_P1
                )
                root_histograms["Gen"]["Events"]["Omega_GEN_SIGNAL_M1"].Fill(
                    gen_w, weight * weight_M1
                )

                # Omega vs cos(theta_rho)
                root_histograms["Reco"]["Events"]["OmegaCosTheta_SIGNAL"].Fill(
                    w, cos_theta_rho, weight
                )
                root_histograms["Reco"]["Events"]["OmegaCosTheta_SIGNAL_P1"].Fill(
                    w, cos_theta_rho, weight * weight_P1
                )
                root_histograms["Reco"]["Events"]["OmegaCosTheta_SIGNAL_M1"].Fill(
                    w, cos_theta_rho, weight * weight_M1
                )

                root_histograms["Gen"]["Events"]["OmegaCosThetaTau_GEN_SIGNAL"].Fill(
                    gen_w, gen_cos_theta_tau, weight
                )
                root_histograms["Gen"]["Events"]["OmegaCosThetaTau_GEN_SIGNAL_P1"].Fill(
                    gen_w, gen_cos_theta_tau, weight * weight_P1
                )
                root_histograms["Gen"]["Events"]["OmegaCosThetaTau_GEN_SIGNAL_M1"].Fill(
                    gen_w, gen_cos_theta_tau, weight * weight_M1
                )

                # CosTheta
                root_histograms["Reco"]["Events"]["CosTheta_SIGNAL"].Fill(
                    cos_theta, weight
                )
                root_histograms["Reco"]["Events"]["CosTheta_SIGNAL_P1"].Fill(
                    cos_theta, weight_M1 * weight
                )
                root_histograms["Reco"]["Events"]["CosTheta_SIGNAL_M1"].Fill(
                    cos_theta, weight_P1 * weight
                )

                root_histograms["Gen"]["Events"]["CosTheta_GEN_SIGNAL"].Fill(
                    cos_theta, weight
                )
                root_histograms["Gen"]["Events"]["CosTheta_GEN_SIGNAL_P1"].Fill(
                    cos_theta, weight_M1 * weight
                )
                root_histograms["Gen"]["Events"]["CosTheta_GEN_SIGNAL_M1"].Fill(
                    cos_theta, weight_P1 * weight
                )

                # CosPsi
                root_histograms["Reco"]["Events"]["CosPsi_SIGNAL"].Fill(
                    cos_psi, weight
                )
                root_histograms["Reco"]["Events"]["CosPsi_SIGNAL_P1"].Fill(
                    cos_psi, weight_M1 * weight
                )
                root_histograms["Reco"]["Events"]["CosPsi_SIGNAL_M1"].Fill(
                    cos_psi, weight_P1 * weight
                )

                root_histograms["Gen"]["Events"]["CosPsi_GEN_SIGNAL"].Fill(
                    gen_cos_psi, weight
                )
                root_histograms["Gen"]["Events"]["CosPsi_GEN_SIGNAL_P1"].Fill(
                    gen_cos_psi, weight_M1 * weight
                )
                root_histograms["Gen"]["Events"]["CosPsi_GEN_SIGNAL_M1"].Fill(
                    gen_cos_psi, weight_P1 * weight
                )

                # Tipos de mesón (GEN vs RECO)
                root_histograms["Matched"]["Events"]["MesonCosTheta"].Fill(
                    math.cos(genMesonTheta), weight
                )
                root_histograms["Matched"]["Events"]["MesonCosTheta_P1"].Fill(
                    math.cos(genMesonTheta), weight_P1 * weight
                )
                root_histograms["Matched"]["Events"]["MesonCosTheta_M1"].Fill(
                    math.cos(genMesonTheta), weight_M1 * weight
                )

                root_histograms["Gen"]["Events"]["MesonType"].Fill(
                    genTauID, weight
                )
                root_histograms["Reco"]["Events"]["RecoMesonType"].Fill(
                    recoTauID, weight
                )

                # RecoMesonE y cosTheta para señal
                root_histograms["Reco"]["Events"]["RecoMesonEOverBeamE_SIGNAL"].Fill(
                    recoMesonE / beamE, weight
                )
                root_histograms["Reco"]["Events"]["RecoMesonEOverBeamE_SIGNAL_P1"].Fill(
                    recoMesonE / beamE, weight_P1 * weight
                )
                root_histograms["Reco"]["Events"]["RecoMesonEOverBeamE_SIGNAL_M1"].Fill(
                    recoMesonE / beamE, weight_M1 * weight
                )

                root_histograms["Reco"]["Events"]["RecoMeson_X"].Fill(
                    x, weight
                )
                root_histograms["Reco"]["Events"]["RecoMeson_X_P1"].Fill(
                    x, weight * weight_P1
                )
                root_histograms["Reco"]["Events"]["RecoMeson_X_M1"].Fill(
                    x, weight * weight_M1
                )

                root_histograms["Reco"]["Events"]["RecoMesonCosTheta_SIGNAL"].Fill(
                    math.cos(recoMesonTheta), weight
                )
                root_histograms["Reco"]["Events"]["RecoMesonCosTheta_SIGNAL_P1"].Fill(
                    math.cos(recoMesonTheta), weight_P1 * weight
                )
                root_histograms["Reco"]["Events"]["RecoMesonCosTheta_SIGNAL_M1"].Fill(
                    math.cos(recoMesonTheta), weight_M1 * weight
                )

            else:
                # Fondo (BG) - misma lógica de categorías que en el análisis original
                root_histograms["Reco"]["Events"]["RecoMesonEOverBeamE_BG"].Fill(
                    recoMesonE / beamE, weight
                )
                root_histograms["Reco"]["Events"]["RecoMesonCosTheta_BG"].Fill(
                    math.cos(recoMesonTheta), weight
                )
                root_histograms["Matched"]["Events"]["MesonEOverBeamE_BG"].Fill(
                    genMesonE / beamE, weight
                )
                root_histograms["Matched"]["Events"]["MesonCosTheta_BG"].Fill(
                    math.cos(genMesonTheta), weight
                )
                root_histograms["Reco"]["Events"]["RecoMeson_X_BG"].Fill(
                    x, weight
                )

                root_histograms["Gen"]["Events"]["MesonType_BG"].Fill(
                    genTauID, weight
                )
                root_histograms["Reco"]["Events"]["RecoMesonType_BG"].Fill(
                    recoTauID, weight
                )

                root_histograms["Reco"]["Events"]["Omega_BG"].Fill(w, weight)
                root_histograms["Reco"]["Events"]["CosTheta_BG"].Fill(
                    cos_theta, weight
                )
                root_histograms["Reco"]["Events"]["CosPsi_BG"].Fill(
                    cos_psi, weight
                )
                root_histograms["Reco"]["Events"]["OmegaCosTheta_BG"].Fill(
                    w, cos_theta_rho, weight
                )
                root_histograms["Gen"]["Events"]["OmegaCosThetaTau_GEN_BG"].Fill(
                    gen_w, gen_cos_theta_tau, weight
                )

                root_histograms["Gen"]["Events"]["Omega_GEN_BG"].Fill(
                    gen_w, weight
                )
                root_histograms["Gen"]["Events"]["CosTheta_GEN_BG"].Fill(
                    gen_cos_theta, weight
                )
                root_histograms["Gen"]["Events"]["CosPsi_GEN_BG"].Fill(
                    gen_cos_psi, weight
                )

                # Sub-categorías de fondo según genTauID
                if genTauID == -13:  # muones
                    root_histograms["Reco"]["Events"]["RecoMesonEOverBeamE_BGMuon"].Fill(
                        recoMesonE / beamE, weight
                    )
                    root_histograms["Matched"]["Events"]["MesonEOverBeamE_BGMuon"].Fill(
                        genMesonE / beamE, weight
                    )
                    root_histograms["Reco"]["Events"]["RecoMesonE_BGMuon"].Fill(
                        recoMesonE, weight
                    )

                    root_histograms["Reco"]["Events"]["RecoMesonCosTheta_BGMuon"].Fill(
                        math.cos(recoMesonTheta), weight
                    )
                    root_histograms["Matched"]["Events"]["MesonCosTheta_BGMuon"].Fill(
                        math.cos(genMesonTheta), weight
                    )

                    root_histograms["Reco"]["Events"]["RecoMeson_BGMuon_PhiTheta"].Fill(
                        recoMesonTheta, recoMesonPhi, weight
                    )
                    root_histograms["Reco"]["Events"]["RecoMeson_BGMuon_PtTheta"].Fill(
                        recoMesonTheta, recoMesonPt, weight
                    )

                    root_histograms["Reco"]["Events"]["Omega_BGMuon"].Fill(
                        w, weight
                    )
                    root_histograms["Reco"]["Events"]["OmegaCosTheta_BGMuon"].Fill(
                        w, cos_theta_rho, weight
                    )
                    root_histograms["Reco"]["Events"]["CosTheta_BGMuon"].Fill(
                        cos_theta, weight
                    )
                    root_histograms["Reco"]["Events"]["CosPsi_BGMuon"].Fill(
                        cos_psi, weight
                    )

                    root_histograms["Gen"]["Events"]["OmegaCosThetaTau_GEN_BGMuon"].Fill(
                        gen_w, gen_cos_theta_tau, weight
                    )
                    root_histograms["Gen"]["Events"]["Omega_GEN_BGMuon"].Fill(
                        gen_w, weight
                    )
                    root_histograms["Gen"]["Events"]["CosTheta_GEN_BGMuon"].Fill(
                        gen_cos_theta, weight
                    )
                    root_histograms["Gen"]["Events"]["CosPsi_GEN_BGMuon"].Fill(
                        gen_cos_psi, weight
                    )

                elif genTauID == -11:  # electrones
                    root_histograms["Reco"]["Events"]["RecoMesonEOverBeamE_BGEle"].Fill(
                        recoMesonE / beamE, weight
                    )
                    root_histograms["Matched"]["Events"]["MesonEOverBeamE_BGEle"].Fill(
                        genMesonE / beamE, weight
                    )
                    root_histograms["Reco"]["Events"]["RecoMesonCosTheta_BGEle"].Fill(
                        math.cos(recoMesonTheta), weight
                    )
                    root_histograms["Matched"]["Events"]["MesonCosTheta_BGEle"].Fill(
                        math.cos(genMesonTheta), weight
                    )

                    root_histograms["Reco"]["Events"]["RecoMeson_BGEle_PhiTheta"].Fill(
                        recoMesonTheta, recoMesonPhi, weight
                    )
                    root_histograms["Reco"]["Events"]["RecoMeson_BGEle_PtTheta"].Fill(
                        recoMesonTheta, recoMesonPt, weight
                    )

                    root_histograms["Reco"]["Events"]["Omega_BGEle"].Fill(
                        w, weight
                    )
                    root_histograms["Reco"]["Events"]["OmegaCosTheta_BGEle"].Fill(
                        w, cos_theta_rho, weight
                    )
                    root_histograms["Reco"]["Events"]["CosTheta_BGEle"].Fill(
                        cos_theta, weight
                    )
                    root_histograms["Reco"]["Events"]["CosPsi_BGEle"].Fill(
                        cos_psi, weight
                    )

                    root_histograms["Gen"]["Events"]["OmegaCosThetaTau_GEN_BGEle"].Fill(
                        gen_w, gen_cos_theta_tau, weight
                    )
                    root_histograms["Gen"]["Events"]["Omega_GEN_BGEle"].Fill(
                        gen_w, weight
                    )
                    root_histograms["Gen"]["Events"]["CosTheta_GEN_BGEle"].Fill(
                        gen_cos_theta, weight
                    )
                    root_histograms["Gen"]["Events"]["CosPsi_GEN_BGEle"].Fill(
                        gen_cos_psi, weight
                    )

                elif genTauID == 0:  # piones
                    root_histograms["Reco"]["Events"]["RecoMesonEOverBeamE_BGPion"].Fill(
                        recoMesonE / beamE, weight
                    )
                    root_histograms["Matched"]["Events"]["MesonEOverBeamE_BGPion"].Fill(
                        genMesonE / beamE, weight
                    )

                    root_histograms["Reco"]["Events"]["RecoMesonCosTheta_BGPion"].Fill(
                        math.cos(recoMesonTheta), weight
                    )
                    root_histograms["Matched"]["Events"]["MesonCosTheta_BGPion"].Fill(
                        math.cos(genMesonTheta), weight
                    )

                    root_histograms["Reco"]["Events"]["Omega_BGPion"].Fill(
                        w, weight
                    )
                    root_histograms["Reco"]["Events"]["OmegaCosTheta_BGPion"].Fill(
                        w, cos_theta_rho, weight
                    )

                    root_histograms["Reco"]["Events"]["CosTheta_BGPion"].Fill(
                        cos_theta, weight
                    )
                    root_histograms["Reco"]["Events"]["CosPsi_BGPion"].Fill(
                        cos_psi, weight
                    )

                    root_histograms["Gen"]["Events"]["OmegaCosThetaTau_GEN_BGPion"].Fill(
                        gen_w, gen_cos_theta_tau, weight
                    )
                    root_histograms["Gen"]["Events"]["Omega_GEN_BGPion"].Fill(
                        gen_w, weight
                    )
                    root_histograms["Gen"]["Events"]["CosTheta_GEN_BGPion"].Fill(
                        gen_cos_theta, weight
                    )
                    root_histograms["Gen"]["Events"]["CosPsi_GEN_BGPion"].Fill(
                        gen_cos_psi, weight
                    )

                elif genTauID == 1:  # rho
                    root_histograms["Reco"]["Events"]["RecoMesonEOverBeamE_BGRho"].Fill(
                        recoMesonE / beamE, weight
                    )
                    root_histograms["Matched"]["Events"]["MesonEOverBeamE_BGRho"].Fill(
                        genMesonE / beamE, weight
                    )

                    root_histograms["Reco"]["Events"]["RecoMesonCosTheta_BGRho"].Fill(
                        math.cos(recoMesonTheta), weight
                    )
                    root_histograms["Matched"]["Events"]["MesonCosTheta_BGRho"].Fill(
                        math.cos(genMesonTheta), weight
                    )

                    root_histograms["Reco"]["Events"]["Omega_BGRho"].Fill(
                        w, weight
                    )
                    root_histograms["Reco"]["Events"]["OmegaCosTheta_BGRho"].Fill(
                        w, cos_theta_rho, weight
                    )

                    root_histograms["Reco"]["Events"]["CosTheta_BGRho"].Fill(
                        cos_theta, weight
                    )
                    root_histograms["Reco"]["Events"]["CosPsi_BGRho"].Fill(
                        cos_psi, weight
                    )

                    root_histograms["Gen"]["Events"]["OmegaCosThetaTau_GEN_BGRho"].Fill(
                        gen_w, gen_cos_theta_tau, weight
                    )
                    root_histograms["Gen"]["Events"]["Omega_GEN_BGRho"].Fill(
                        gen_w, weight
                    )
                    root_histograms["Gen"]["Events"]["CosTheta_GEN_BGRho"].Fill(
                        gen_cos_theta, weight
                    )
                    root_histograms["Gen"]["Events"]["CosPsi_GEN_BGRho"].Fill(
                        gen_cos_psi, weight
                    )

                elif genTauID == 10:  # a1
                    root_histograms["Reco"]["Events"]["RecoMesonEOverBeamE_BGA1"].Fill(
                        recoMesonE / beamE, weight
                    )
                    root_histograms["Matched"]["Events"]["MesonEOverBeamE_BGA1"].Fill(
                        genMesonE / beamE, weight
                    )

                    root_histograms["Reco"]["Events"]["RecoMesonCosTheta_BGA1"].Fill(
                        math.cos(recoMesonTheta), weight
                    )
                    root_histograms["Matched"]["Events"]["MesonCosTheta_BGA1"].Fill(
                        math.cos(genMesonTheta), weight
                    )

                    root_histograms["Reco"]["Events"]["Omega_BGA1"].Fill(
                        w, weight
                    )
                    root_histograms["Reco"]["Events"]["OmegaCosTheta_BGA1"].Fill(
                        w, cos_theta_rho, weight
                    )

                    root_histograms["Reco"]["Events"]["CosTheta_BGA1"].Fill(
                        cos_theta, weight
                    )
                    root_histograms["Reco"]["Events"]["CosPsi_BGA1"].Fill(
                        cos_psi, weight
                    )

                    root_histograms["Gen"]["Events"]["OmegaCosThetaTau_GEN_BGA1"].Fill(
                        gen_w, gen_cos_theta_tau, weight
                    )
                    root_histograms["Gen"]["Events"]["Omega_GEN_BGA1"].Fill(
                        gen_w, weight
                    )
                    root_histograms["Gen"]["Events"]["CosTheta_GEN_BGA1"].Fill(
                        gen_cos_theta, weight
                    )
                    root_histograms["Gen"]["Events"]["CosPsi_GEN_BGA1"].Fill(
                        gen_cos_psi, weight
                    )

                else:  # other BG
                    root_histograms["Reco"]["Events"]["RecoMesonEOverBeamE_BGOther"].Fill(
                        recoMesonE / beamE, weight
                    )
                    root_histograms["Matched"]["Events"]["MesonEOverBeamE_BGOther"].Fill(
                        genMesonE / beamE, weight
                    )

                    root_histograms["Reco"]["Events"]["RecoMesonCosTheta_BGOther"].Fill(
                        math.cos(recoMesonTheta), weight
                    )
                    root_histograms["Matched"]["Events"]["MesonCosTheta_BGOther"].Fill(
                        math.cos(genMesonTheta), weight
                    )

                    root_histograms["Reco"]["Events"]["Omega_BGOther"].Fill(
                        w, weight
                    )
                    root_histograms["Reco"]["Events"]["OmegaCosTheta_BGOther"].Fill(
                        w, cos_theta_rho, weight
                    )

                    root_histograms["Reco"]["Events"]["CosTheta_BGOther"].Fill(
                        cos_theta, weight
                    )
                    root_histograms["Reco"]["Events"]["CosPsi_BGOther"].Fill(
                        cos_psi, weight
                    )

                    root_histograms["Gen"]["Events"]["OmegaCosThetaTau_GEN_BGOther"].Fill(
                        gen_w, gen_cos_theta_tau, weight
                    )
                    root_histograms["Gen"]["Events"]["Omega_GEN_BGOther"].Fill(
                        gen_w, weight
                    )
                    root_histograms["Gen"]["Events"]["CosTheta_GEN_BGOther"].Fill(
                        gen_cos_theta, weight
                    )
                    root_histograms["Gen"]["Events"]["CosPsi_GEN_BGOther"].Fill(
                        gen_cos_psi, weight
                    )


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
    logger_io.info(f"All histograms written and file closed. Results in {fileOutName}")


if __name__ == "__main__":
    main()
