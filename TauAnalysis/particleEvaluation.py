import argparse
import copy
import logging
import math
import os
import pickle
import pprint
import sys
from array import array
from pathlib import Path

import edm4hep
import numpy as np
import pandas as pd
import ROOT
import yaml
from podio import root_io
from ROOT import TH1F, TH2F, TFile, TTree

from modules import (ParticleObjects, electronReco, muonReco, myutils, pi0Reco,
                     tauReco, particleMatch)
from modules.particleMatch import ParticleMatchResults
from modules.ParticleObjects import GenRecoMatched,RecoParticle, GenParticle, Track
c_light = 2.99792458e8
Bz_clic = 4.0
Bz_cld = 2.0
mchp = 0.139570

def omega_to_pt(omega, isclic):
    if isclic:
        Bz = Bz_clic
    else:
        Bz = Bz_cld
    a = c_light * 1e3 * 1e-15
    return a * Bz / abs(omega)

def track_momentum(trackstate, isclic=True):
    pt = omega_to_pt(trackstate.omega, isclic)
    phi = trackstate.phi
    pz = trackstate.tanLambda * pt
    px = pt * math.cos(phi)
    py = pt * math.sin(phi)
    p = math.sqrt(px * px + py * py + pz * pz)
    energy = math.sqrt(p * p + mchp * mchp)
    theta = math.acos(pz / p)
    # print(p, theta, phi, energy)
    return p, theta, phi, energy, px, py, pz

# Load config (necessary for set up the logger)
default_config = "config/default/taurecolong.yaml"
# Output Configuration
outputbasepath = "Results/TauReco/"

general_configs = myutils.setup_analysis_config(default_config, outputbasepath, particle_analysis=True)


loggers = general_configs["loggers"]

run_config = general_configs["config"]

# config = myutils.load_yaml_config(args.config, default_config)


# Cut Configuration
dRMax=run_config["cuts"]["dRMax"]
minPTauPhoton =run_config["cuts"]["TauPhotonPCut"]
minPTauPion = run_config["cuts"]["TauPionPCut"]
PNeutron = run_config["cuts"]["NeutronCut"]
dRMatch = run_config["cuts"]["MatchedGenMinDR"]
generalPCut = run_config["cuts"]["generalPCut"]

selectDecay=general_configs["decay"]

outputpath = general_configs["outputpath"]
# # obtener ruta de la última carpeta del outputpath
# last_folder = Path(outputpath).parts[-1]
# last_folder = "ParticleEval_" + last_folder
# new_path = Path(outputpath)
# new_path = Path(*new_path.parts[:-1], last_folder)
# general_configs["outputpath"] = str(new_path)
fileOutName = os.path.join(general_configs["outputpath"], general_configs["fileOutName"])

logger_config = loggers["config"]
logger_io = loggers["io"]
logger_process = loggers["processing"]
logger_pi0mass = loggers["pi0mass"]
logger_io.info("Output path: %s", fileOutName)
args = general_configs["args"]
print("ARGUMENTOS args:", args)
# Continue with the rest of configs

# ------------------------------------------------------------------------
# General Configuration
sample=run_config["general"]["sample"]
matched_cm_arg = general_configs["flags"]["matched_cm"]
test_arg = general_configs["flags"]["test"]

logger_config.info("Configuration loaded!")
logger_config.info("Configuration:\n%s", pprint.pformat(general_configs, indent=4))


# ------------------------------------------------------------------------
gatr_results_path = general_configs["args"].gatr_result

filenames, mlpf_results = myutils.get_root_trees_path(sample, gatr_results_path, loggers,args.test, args)
reader = root_io.Reader(filenames)

genparts = "MCParticles"
pfobjects = "PandoraPFOs"
track_column = "SiTracks_Refitted"
track_links_column = "SiTracksMCTruthLink"

histogram_config = general_configs.get("histograms_config", {})
root_histograms = myutils.set_up_root_histograms(histogram_config)
miss_matched_histograms_p = dict()
miss_matched_histograms_theta = dict()

confusion_events = {"Event":[], "Cluster":[], "Cf_w_e" : [], "Cf_w_mu": [], "Cf_w_n": []}

# print(root_histograms)
# exit(0)
countEvents = 0
tracks = True
for eventid, event in enumerate(reader.get("events")):
    logger_process.debug("Processing event %d", eventid)
    if countEvents % 1000 == 0:
        logger_process.info("Processing event %d", countEvents)
    countEvents += 1

    mc_particles = event.get(genparts)
    pfos = event.get(pfobjects)


    genTaus = tauReco.findAllGenTaus(mc_particles)
    nGenTaus = len(genTaus)

    logger_process.debug(
        "Found %d gen taus. Details:\n%s",
        nGenTaus,
        "\n".join("GenTau %d: %s" % (i, tau) for i, tau in genTaus.items()),
    )
    if gatr_results_path is not None and not general_configs["args"].test_pfo:
        logger_process.debug("Using GATr results for event %d", eventid)
        particles = mlpf_results.get(eventid, {})
        recoTau = tauReco.findAllTaus(
            particles, dRMax, minPTauPhoton, minPTauPion, PNeutron, generalPCut, charge_condition=False
        )
        recoElectrons = electronReco.findAllElectrons(particles, generalPCut)
        recoMuons = muonReco.findAllMuons(particles, generalPCut)  
    else:
        logger_process.debug("Using PandoraPFO results for event %d", eventid)
        
        particles = pfos
        recoTau = tauReco.findAllTaus(
            particles, dRMax, minPTauPhoton, minPTauPion, PNeutron, generalPCut
        )
        recoElectrons = electronReco.findAllElectrons(particles, generalPCut)
        recoMuons = muonReco.findAllMuons(particles, generalPCut)


  
    # Pion Matching
    # if gatr_results_path is not None and not general_configs["args"].test_pfo:
    try:
        matching_results = particleMatch.MatchedUnmatchedParticles(mc_particles,
                                                                particles,
                                                                maxDRMatch=1,
                                                                non_considered_particles = [12, 14, 16],
                                                                force=False)
        pdg_code = 211
        matching_pions: ParticleMatchResults = matching_results[str(pdg_code)]
        # root_histograms["Reco"]["Events"]["DuplicatePionMatch"].Fill(n_pions_reco_duplicated)
        # matching_photons = matching_results['photons']
        # unmatched_gen_photons = matching_photons["unmatched_gen_photons"]
        # unmatched_reco_photons = matching_photons["unmatched_reco_photons"]
        # gen_photons_matched_with_reco_photons = matching_photons["gen_photons_matched_with_reco_photons"]
        # reco_photons_matched_with_other_particles = matching_photons["reco_photons_matched_with_other_particles"]
        # gen_photons_matched_with_other_particles = matching_photons["gen_photons_matched_with_other_particles"]

            
    except Exception as e:

        logger_process.warning(f"Error reading {eventid}: {e}")
        matching_pions = ParticleMatchResults()

    # ------------------------------------------------------------------------
    # TRACK RECONSTRUCTION (SiTracks)
    # ------------------------------------------------------------------------
    try:
        tracks = event.get(track_column)
        track_links = event.get(track_links_column)

        logger_process.debug(f"Found {len(tracks)} tracks in event {eventid}")

        isclic = False  # o detectarlo desde config si lo tienes definido en YAML

    except Exception as e:
        if tracks:
            logger_process.warning(f"Error reading tracks in event {eventid}: {e}")
        tracks = False
        
        track_links = []
        # unmatched_gen_photons = {}
        # unmatched_reco_photons = {}
        # gen_photons_matched_with_reco_photons = {}
        # reco_photons_matched_with_other_particles = {}
        # gen_photons_matched_with_other_particles = {}
        
    logger_process.debug(
        "Total Gen Pions: %d, Total Reco Pions: %d, Total Unmatched Gen Pions: %d, Total Unmatched Reco Pions: %d, Matched Pions: %d",
        len(matching_pions.all_gen), len(matching_pions.all_reco), len(matching_pions.unmatched_gen), len(matching_pions.unmatched_reco), len(matching_pions.gen_matched_with_reco)
    )
    # print(matching_pions.gen_matched_with_reco)
    # for gen_id, gen_pion in matching_pions.all_gen.items():
    #     print(gen_id, gen_pion)
    #     print(gen_pion in matching_pions.gen_matched_with_reco)
    # exit(0)
    # all_pions = particleMatch.GetGenPions(mc_particles, False)
    # logger_process.debug("All gen pions: %d", len(all_pions))
    # if len(matching_pions.all_gen) != len(all_pions):
    #     logger_process.warning(
    #         "MISSMATCH IN GEN PION COUNT: matched %d vs all final state %d",
    #         len(matching_pions.all_gen), len(all_pions)
    #     )
    # Look for clústers of pions in data
    pion_clusters = []
    pion_clusters_max_distance = {}
    distances_matrix = np.zeros((len(matching_pions.all_gen), len(matching_pions.all_gen)))
    for idx_i, (pion_id, pion) in enumerate(matching_pions.all_gen.items()):
        cluster = set([pion_id])
        for idx_j, (other_pion_id, other_pion) in enumerate(matching_pions.all_gen.items()):
            if pion_id == other_pion_id:
                continue
            dR = myutils.dRAngle(pion.getvisMomentum(), other_pion.getMomentum())
            distances_matrix[idx_i, idx_j] = dR
            if dR < 0.5:
                cluster.add(other_pion_id)
        if len(cluster) > 1 and cluster not in pion_clusters:
            pion_clusters.append(cluster)
    # Check subsets and remove them
    cleaned_clusters = copy.deepcopy(pion_clusters)
    for cluster in pion_clusters:
        for other_cluster in pion_clusters:
            if cluster == other_cluster:
                continue
            if cluster.issubset(other_cluster):
                cleaned_clusters.remove(cluster)
                break
    
    for cluster in cleaned_clusters:
        max_distance = 0
        for pion_id in cluster:
            pion = matching_pions.all_gen[pion_id]
            # TODO revisar angulo por trazas
            # mc = pion.mcp             
            # associations = [link for link in track_links if link.getSim() == mc]
            # best_assoc = max(associations, key=lambda a: a.getWeight())
            # track = best_assoc.getRec()
            # trackstate = track.getTrackStates()[0]
            # p, theta, phi, energy, px, py, pz = track_momentum(trackstate, isclic=isclic)
            # pion_track_p4 = ROOT.TLorentzVector()
            # pion_track_p4.SetPxPyPzE(px, py, pz, energy)
            
            for other_pion_id in cluster:
                if pion_id == other_pion_id:
                    continue
                other_pion = matching_pions.all_gen[other_pion_id]
                dR = myutils.dRAngle(pion.getMomentum(), other_pion.getMomentum())
                if dR > max_distance:
                    max_distance = dR
        pion_clusters_max_distance[frozenset(cluster)] = max_distance

    logger_process.debug(f"Found {len(pion_clusters)} pion clusters in event {eventid}: {cleaned_clusters}")
    logger_process.debug(f"Distances matrix:\n{distances_matrix}")

    for pion_id, reco_pion in matching_pions.all_reco.items():
        root_histograms["Reco"]["Events"]["AllPionsP"].Fill(reco_pion.getMomentum().P())
        root_histograms["Reco"]["Events"]["AllPionsTheta"].Fill(reco_pion.getMomentum().Theta())
    
    for pion_id, reco_pion in matching_pions.unmatched_reco.items():
        root_histograms["Matched"]["Events"]["UnmatchedRecoPionsP"].Fill(reco_pion.getMomentum().P())
        root_histograms["Matched"]["Events"]["UnmatchedRecoPionsTheta"].Fill(reco_pion.getMomentum().Theta())

    
    for pion_id, pion in matching_pions.all_gen.items():
        root_histograms["Gen"]["Events"]["AllPionsP"].Fill(pion.getMomentum().P())
        root_histograms["Gen"]["Events"]["AllPionsTheta"].Fill(pion.getMomentum().Theta())
        pion_in_cluster = False
        for cluster in cleaned_clusters:
            if pion_id in cluster:
                pion_in_cluster = True
                break
        if pion_in_cluster:
            # logger_process.debug(f"Gen pion {pion_id} is in a cluster")
            root_histograms["Gen"]["Events"]["ClusterPionP"].Fill(pion.getMomentum().P())
            root_histograms["Gen"]["Events"]["ClusterPionTheta"].Fill(pion.getMomentum().Theta())
        else:
            root_histograms["Gen"]["Events"]["IsolatedPionP"].Fill(pion.getMomentum().P())
            root_histograms["Gen"]["Events"]["IsolatedPionTheta"].Fill(pion.getMomentum().Theta())

    for match_obj in matching_pions.gen_matched_with_reco:
        genPion = match_obj.getGenParticle()
        genPion_id = match_obj.getGenID()
        recoPion = match_obj.getMatchedRecoParticle()
        recoPion_id = match_obj.getMatchedRecoID()
        root_histograms["Matched"]["Events"]["AllPionsGenP"].Fill(genPion.getMomentum().P())
        root_histograms["Matched"]["Events"]["AllPionsGenTheta"].Fill(genPion.getMomentum().Theta())
        root_histograms["Matched"]["Events"]["AllPionsRecoP"].Fill(recoPion.getMomentum().P())
        root_histograms["Matched"]["Events"]["AllPionsRecoTheta"].Fill(recoPion.getMomentum().Theta())
        pion_in_cluster = False
        for cluster in cleaned_clusters:
            if genPion_id in cluster:
                pion_in_cluster = True
                break
        if pion_in_cluster:
            # logger_process.debug(f"Gen pion {genPion_id} matched with reco pion {recoPion_id} is in a cluster")
            root_histograms["Matched"]["Events"]["ClusterPionGenP"].Fill(genPion.getMomentum().P())
            root_histograms["Matched"]["Events"]["ClusterPionGenTheta"].Fill(genPion.getMomentum().Theta())
            root_histograms["Matched"]["Events"]["ClusterPionRecoP"].Fill(recoPion.getMomentum().P())
            root_histograms["Matched"]["Events"]["ClusterPionRecoTheta"].Fill(recoPion.getMomentum().Theta())
        else:
            root_histograms["Matched"]["Events"]["IsolatedPionGenP"].Fill(genPion.getMomentum().P())
            root_histograms["Matched"]["Events"]["IsolatedPionGenTheta"].Fill(genPion.getMomentum().Theta())
            root_histograms["Matched"]["Events"]["IsolatedPionRecoP"].Fill(recoPion.getMomentum().P())
            root_histograms["Matched"]["Events"]["IsolatedPionRecoTheta"].Fill(recoPion.getMomentum().Theta())
    #     "Unmatched Gen Photons: %d, Unmatched Reco Photons: %d",
    
    for pion_id, gen_pion in matching_pions.unmatched_gen.items():
        root_histograms["Matched"]["Events"]["GenPionsUnmatchedGenP"].Fill(gen_pion.getMomentum().P())
        root_histograms["Matched"]["Events"]["GenPionsUnmatchedGenTheta"].Fill(gen_pion.getMomentum().Theta())
        pion_in_cluster = False
        for cluster, max_distance in pion_clusters_max_distance.items():
            if pion_id in cluster:
                pion_in_cluster = True
                break
        if pion_in_cluster:
            # logger_process.debug(f"Unmatched gen pion {pion_id} is in a cluster")
            root_histograms["Matched"]["Events"]["ClusterPionsUnmatchedGenP"].Fill(gen_pion.getMomentum().P())
            root_histograms["Matched"]["Events"]["ClusterPionsUnmatchedGenTheta"].Fill(gen_pion.getMomentum().Theta())
            root_histograms["Matched"]["Events"]["ClusterUnmatchedGenPionDr"].Fill(max_distance)
        else:
            root_histograms["Matched"]["Events"]["IsolatedPionsUnmatchedGenP"].Fill(gen_pion.getMomentum().P())
            root_histograms["Matched"]["Events"]["IsolatedPionsUnmatchedGenTheta"].Fill(gen_pion.getMomentum().Theta())

    for match_obj in matching_pions.reco_matched_with_other:
        reco_pion = match_obj.getMatchedRecoParticle()
        gen_part = match_obj.getGenParticle()
        gen_part_PDG = match_obj.getGenPDG()
        # root_histograms["Matched"]["Events"]["MissmatchedRecoPionsRecoP"].Fill(reco_pion.getMomentum().P())
        if abs(gen_part_PDG)==11:
            root_histograms["Matched"]["Events"]["RecoPions_BG_e"].Fill(reco_pion.getMomentum().P())
        elif abs(gen_part_PDG)==13:
            root_histograms["Matched"]["Events"]["RecoPions_BG_mu"].Fill(reco_pion.getMomentum().P())
        elif abs(gen_part_PDG)==2112:
            root_histograms["Matched"]["Events"]["RecoPions_BG_n"].Fill(reco_pion.getMomentum().P())
        else:
            root_histograms["Matched"]["Events"]["RecoPions_BG_other"].Fill(reco_pion.getMomentum().P())
    for match_obj in matching_pions.gen_matched_with_other:
        confusion_events["Event"].append(eventid)
        genPion = match_obj.getGenParticle()
        genPion_id = match_obj.getGenID()
        recoParticle = match_obj.getMatchedRecoParticle()
        recoPion_pdg = match_obj.getMatchedRecoPDG()
        root_histograms["Matched"]["Events"]["MissmatchedGenPionsGenP"].Fill(genPion.getMomentum().P())
        root_histograms["Matched"]["Events"]["MissmatchedGenPionsGenTheta"].Fill(genPion.getMomentum().Theta())
        if f"General_{recoPion_pdg}P" not in root_histograms["Matched"]["Events"]:
            root_histograms["Matched"]["Events"][f"General_{recoPion_pdg}P"] = TH1F(f"hMissmatchedGenPionP_{recoPion_pdg}","", 100, 0, 50)
            root_histograms["Matched"]["Events"][f"General_{recoPion_pdg}Theta"] = TH1F(f"hMissmatchedGenPionTheta_{recoPion_pdg}","", 100, 0, 3.15)
            root_histograms["Matched"]["Events"][f"Isolated_{recoPion_pdg}P"] = TH1F(f"hMissmatchedIsolatedGenPionP_{recoPion_pdg}","", 100, 0, 50)
            root_histograms["Matched"]["Events"][f"Isolated_{recoPion_pdg}Theta"] = TH1F(f"hMissmatchedIsolatedGenPionTheta_{recoPion_pdg}","", 100, 0, 3.15)
            root_histograms["Matched"]["Events"][f"Cluster_{recoPion_pdg}P"] = TH1F(f"hMissmatchedClusterGenPionP_{recoPion_pdg}","", 100, 0, 50)
            root_histograms["Matched"]["Events"][f"Cluster_{recoPion_pdg}Theta"] = TH1F(f"hMissmatchedClusterGenPionTheta_{recoPion_pdg}","", 100, 0, 3.15)
            root_histograms["Matched"]["Events"][f"Cluster_{recoPion_pdg}DR"] = TH1F(f"hMissmatchedClusterGenPionDR_{recoPion_pdg}","", 60, 0, 0.5)
        root_histograms["Matched"]["Events"][f"General_{recoPion_pdg}P"].Fill(genPion.getMomentum().P())
        root_histograms["Matched"]["Events"][f"General_{recoPion_pdg}Theta"].Fill(genPion.getMomentum().Theta())
        pion_in_cluster = False
        
        for cluster, max_distance in pion_clusters_max_distance.items():
            if genPion_id in cluster:
                pion_in_cluster = True
                break
        confusion_events["Cluster"].append(cluster)
        if recoPion_pdg == 11:
            confusion_events["Cf_w_e"].append(1)
        else:
            confusion_events["Cf_w_e"].append(0)
        if recoPion_pdg == 13:
            confusion_events["Cf_w_mu"].append(1)
        else:
            confusion_events["Cf_w_mu"].append(0)
        if recoPion_pdg == 2112:
            confusion_events["Cf_w_n"].append(1)
        else:
            confusion_events["Cf_w_n"].append(0)
            
        if pion_in_cluster:
            # logger_process.debug(f"Gen pion {genPion_id} matched with reco particle {recoPion_id} is in a cluster")
            root_histograms["Matched"]["Events"]["MissmatchedClusterPionsGenP"].Fill(genPion.getMomentum().P())
            root_histograms["Matched"]["Events"]["MissmatchedClusterPionsGenTheta"].Fill(genPion.getMomentum().Theta())
            root_histograms["Matched"]["Events"][f"Cluster_{recoPion_pdg}P"].Fill(genPion.getMomentum().P())
            root_histograms["Matched"]["Events"][f"Cluster_{recoPion_pdg}Theta"].Fill(genPion.getMomentum().Theta())
            root_histograms["Matched"]["Events"][f"Cluster_{recoPion_pdg}DR"].Fill(max_distance)
        else:
            root_histograms["Matched"]["Events"]["MissmatchedIsolatedPionsGenP"].Fill(genPion.getMomentum().P())
            root_histograms["Matched"]["Events"]["MissmatchedIsolatedPionsGenTheta"].Fill(genPion.getMomentum().Theta())
            root_histograms["Matched"]["Events"][f"Isolated_{recoPion_pdg}P"].Fill(genPion.getMomentum().P())
            root_histograms["Matched"]["Events"][f"Isolated_{recoPion_pdg}Theta"].Fill(genPion.getMomentum().Theta())

    for clust, max_distance in pion_clusters_max_distance.items():
        pions_in_cluster = [matching_pions.all_gen[pion_id] for pion_id in clust]
        max_momentum = max(pion.getMomentum().P() for pion in pions_in_cluster)
        min_momentum = min(pion.getMomentum().P() for pion in pions_in_cluster)
        n_matched = sum(1 for pion in pions_in_cluster if pion in matching_pions.gen_matched_with_reco)
        for i in range(len(clust)):
            root_histograms["Gen"]["Events"]["GenPionClusterDr"].Fill(max_distance)
            root_histograms["Gen"]["Events"]["ClusteredPionsMaxPvsDR"].Fill(max_momentum, max_distance)
            root_histograms["Gen"]["Events"]["ClusteredPionsMinPvsDR"].Fill(min_momentum, max_distance)  
        for i in range(n_matched):
            root_histograms["Matched"]["Events"]["GenPionClusterDr"].Fill(max_distance)
        for pion in pions_in_cluster:
            if pion in matching_pions.gen_matched_with_reco:
                root_histograms["Matched"]["Events"]["ClusteredMatchedPionsPvsDR"].Fill(pion.getMomentum().P(), max_distance)
                root_histograms["Matched"]["Events"]["ClusteredMatchedPionsThetavsDR"].Fill(pion.getMomentum().Theta(), max_distance)
            else:
                root_histograms["Matched"]["Events"]["ClusteredUnmatchedPionsPvsDR"].Fill(pion.getMomentum().P(), max_distance)
                root_histograms["Matched"]["Events"]["ClusteredUnmatchedPionsThetavsDR"].Fill(pion.getMomentum().Theta(), max_distance)
        # Check efficiency on cluster depending on distance
        
    # len(gen_photons_matched_with_reco_photons) + len(unmatched_gen_photons),
    #     len(gen_photons_matched_with_reco_photons) + len(reco_photons_matched_with_other_particles) + len(unmatched_reco_photons),
    #     len(gen_photons_matched_with_reco_photons),
    #     len(unmatched_gen_photons),
    #     len(unmatched_reco_photons),
    # )
    
    for gen_id, gen_pion in matching_pions.all_gen.items():
        pdg = gen_pion.getPDG()
        mc_p4 = gen_pion.getMomentum()
        # if abs(pdg) != 211:
        #     continue  # solo piones
        # if mc.getGeneratorStatus() != 1:
        #     continue  # solo partículas finales
        # if abs(math.cos(mc_p4.Theta())) > 0.95:
        #     continue  # Cambio para código María
        mc = gen_pion.mcp

        # Buscar asociaciones donde este MC es la partícula simulada
        associations = [link for link in track_links if link.getSim() == mc]
        if not associations:
            root_histograms["Matched"]["Events"]["GenPionsWOTrackGenP"].Fill(mc_p4.P())
            root_histograms["Matched"]["Events"]["GenPionsWOTrackGenTheta"].Fill(mc_p4.Theta())
            # Hay pion reconstruido asociado
            if gen_pion in matching_pions.gen_matched_with_reco:
                root_histograms["Matched"]["Events"]["GenPionsWOTrackWRecoPionGenP"].Fill(mc_p4.P())
                root_histograms["Matched"]["Events"]["GenPionsWOTrackWRecoPionGenTheta"].Fill(mc_p4.Theta())
                idx = matching_pions.gen_matched_with_reco.index(gen_pion)
                matched_reco = matching_pions.gen_matched_with_reco[idx].getMatchedRecoParticle()
                recoP4 = matched_reco.getMomentum()
                root_histograms["Matched"]["Events"]["GenPionsWOTrackWRecoPionRecoP"].Fill(recoP4.P())
                root_histograms["Matched"]["Events"]["GenPionsWOTrackWRecoPionRecoTheta"].Fill(recoP4.Theta())
            else:
                root_histograms["Matched"]["Events"]["GenPionsWOTrackWORecoPionGenP"].Fill(mc_p4.P())
                root_histograms["Matched"]["Events"]["GenPionsWOTrackWORecoPionGenTheta"].Fill(mc_p4.Theta())
            continue
        # if associations:
        root_histograms["Matched"]["Events"]["GenPionsWTrackGenP"].Fill(mc_p4.P())
        root_histograms["Matched"]["Events"]["GenPionsWTrackGenTheta"].Fill(mc_p4.Theta())

        best_assoc = max(associations, key=lambda a: a.getWeight())
        track = best_assoc.getRec()
        trackstate = track.getTrackStates()[0]
        p, theta, phi, energy, px, py, pz = track_momentum(trackstate, isclic=isclic)
        track_p4 = ROOT.TLorentzVector()
        track_p4.SetPxPyPzE(px, py, pz, energy)
        # Trazas asociadas con gen
        root_histograms["Matched"]["Events"]["GenPionsWTrackTrackP"].Fill(track_p4.P())
        root_histograms["Matched"]["Events"]["GenPionsWTrackTrackTheta"].Fill(track_p4.Theta())
        for cluster, max_distance in pion_clusters_max_distance.items():
            if gen_id in cluster:
                root_histograms["Matched"]["Events"]["TrackPionClusterDr"].Fill(max_distance)
                break
        if gen_pion in matching_pions.gen_matched_with_reco:
            root_histograms["Matched"]["Events"]["GenPionsWTrackWRecoGenP"].Fill(mc_p4.P())
            root_histograms["Matched"]["Events"]["GenPionsWTrackWRecoGenTheta"].Fill(mc_p4.Theta())
            # Get index of the matched reco pion
            idx = matching_pions.gen_matched_with_reco.index(gen_pion)
            matched_reco = matching_pions.gen_matched_with_reco[idx].getMatchedRecoParticle()
            recoP4 = matched_reco.getMomentum()
            root_histograms["Matched"]["Events"]["GenPionsWTrackWRecoPionTrackP"].Fill(track_p4.P())
            root_histograms["Matched"]["Events"]["GenPionsWTrackWRecoPionTrackTheta"].Fill(track_p4.Theta())
            root_histograms["Matched"]["Events"]["GenPionsWTrackWRecoPionRecoP"].Fill(recoP4.P())
            root_histograms["Matched"]["Events"]["GenPionsWTrackWRecoPionRecoTheta"].Fill(recoP4.Theta())
        else:
            root_histograms["Matched"]["Events"]["GenPionsWTrackWORecoGenP"].Fill(mc_p4.P())
            root_histograms["Matched"]["Events"]["GenPionsWTrackWORecoGenTheta"].Fill(mc_p4.Theta())
            root_histograms["Matched"]["Events"]["GenPionsWTrackWORecoTrackP"].Fill(track_p4.P())
            root_histograms["Matched"]["Events"]["GenPionsWTrackWORecoTrackTheta"].Fill(track_p4.Theta())

    # if eventid >= 10 and test_arg:
    #     exit(0)  # para pruebas rápidas
    # efficiency = n_matched_pions / n_total_pions if n_total_pions > 0 else 0
    # print(f"Eficiencia de reconstrucción de piones: {efficiency:.2%}")

logger_io.info("Processed %d events", countEvents)
myutils.write_plot_config(root_histograms, outputpath)
# exit(0)
root_histograms = myutils.calc_efficiency(root_histograms, histogram_config)
outfile = ROOT.TFile(fileOutName, "RECREATE")
myutils.write_histograms_recursive(root_histograms)
outfile.Close()
output_config_file = outputpath + "config.yaml"
with open(output_config_file, "w") as file:
    yaml.dump(run_config, file)
    logger_io.info("Configuration file saved to %s", output_config_file)
output_conf_file = outputpath + "confusion_events.csv"
df_confusion = pd.DataFrame(confusion_events)
df_confusion.to_csv(output_conf_file, index=False)
logger_io.info("Confusion events saved to %s", output_conf_file)
logger_io.info("Output written to %s", fileOutName)
  