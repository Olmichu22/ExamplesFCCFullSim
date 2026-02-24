
# bash-5.1$ ls /afs/ciemat.es/user/c/cepeda/public/tausFullSim/tracksAndPF.py 
# /afs/ciemat.es/user/c/cepeda/public/tausFullSim/tracksAndPF.py
# bash-5.1$ cat /afs/ciemat.es/user/c/cepeda/public/tausFullSim/tracksAndPF.py 
# import sys, os, math 
# from array import array
# import numpy as np
# import edm4hep
# import ROOT 

# c_light = 2.99792458e8
# Bz_clic = 4.0
# Bz_cld = 2.0
# mchp = 0.139570
# isclic=False

# def sign(num):
#     return -1 if num < 0 else 1

# def omega_to_pt(omega, isclic):
#     if isclic:
#         Bz = Bz_clic
#     else:
#         Bz = Bz_cld
#     a = c_light * 1e3 * 1e-15
#     return a * Bz / abs(omega)



# def track_momentum(trackstate, isclic=True):
#     pt = omega_to_pt(trackstate.omega, isclic)
#     phi = trackstate.phi
#     pz = trackstate.tanLambda * pt
#     px = pt * math.cos(phi)
#     py = pt * math.sin(phi)
#     p = math.sqrt(px * px + py * py + pz * pz)
#     energy = math.sqrt(p * p + mchp * mchp)
#     theta = math.acos(pz / p)
#     # print(p, theta, phi, energy)
#     return p, theta, phi, energy, px, py, pz


# def find_pandora_pfo_track(hit_index, hit_collection, pfo_collection):
#     pandora_pfo_index = -1
#     for index_pfo, pfo in enumerate(pfo_collection):
#         tracks_pfo = pfo.getTracks()
#         for index_t, track in enumerate(tracks_pfo):
#             # print("index cluster ", index_c)
#             track_index = track.getObjectID().index
#             track_collection_id = track.getObjectID().collectionID
#             if hit_index == track_index and track_collection_id == hit_collection:
#                 pandora_pfo_index = index_pfo
#                 break
#             else:
#                 pandora_pfo_index = -1
#         if pandora_pfo_index >= 0:
#             break
#     return pandora_pfo_index

# def find_mcMatch_track(track_index, mctruthtracks):
#     mc_index = -1
#     for i, link in enumerate(mctruthtracks):
#         rec_ = link.getRec()
#         object_id = rec_.getObjectID().index
#         if object_id==track_index:
#            sim_=link.getSim()
#            mc_index = sim_.getObjectID().index
#         if mc_index  >= 0:
#             break
#     return mc_index 

# def find_track_mcMatch(gen_index, mctruthtracks):
#     track_index = -1
#     for i, link in enumerate(mctruthtracks):
#         sim_ = link.getSim()
#         object_id = sim_.getObjectID().index
#         if object_id==gen_index:
#            rec_=link.getRec()
#            track_index = rec_.getObjectID().index
#         if track_index  >= 0:
#             break
#     return track_index

# def find_pf_mcMatch(gen_index, pftruthtracks):
#     pf_index = -1
#     for i, link in enumerate(pftruthtracks):
#         sim_ = link.getSim()
#         object_id = sim_.getObjectID().index
#         if object_id==gen_index:
#            rec_=link.getRec()
#            pf_index = rec_.getObjectID().index
#         if pf_index  >= 0:
#             break
#     return pf_index

# def recoverTrack(track,pfobjects):
#         trackstate = track.getTrackStates()[0]
#         track_index = track.getObjectID().index
#         isclic=False
#         track_mom = track_momentum(trackstate, isclic)

#         pandora_index_pfo=find_pandora_pfo_track(
#                 track_index,
#                 track.getObjectID().collectionID,
#                 pfobjects,
#             )

#         recoverP4=ROOT.TLorentzVector()
#         pdgID=0
#         charge=sign(trackstate.omega)

#         if pandora_index_pfo!=-1:
#           pf=pfobjects[pandora_index_pfo]
#           pdgID=pf.getPDG()
#           recoverP4.SetXYZM(pf.getMomentum().x,pf.getMomentum().y,pf.getMomentum().z,pf.getMass())
#           charge=pf.getCharge()
#         else:
#           pdgID=211*charge# I could refine this removing muons/electrons ?
#           recoverP4.SetXYZM(track_mom[4],track_mom[5],track_mom[5],0.131)

#         return (pdgID,recoverP4,charge,pandora_index_pfo)


# def recoverTracks (track_coll,pfobjects):

#     P4Recover={}
#     PDGIDRecover={}
#     ChargeRecover={}
#     PFIndexRecover={}

#     countTracks=0

#     for index_t,track in enumerate(track_coll):
#         countTracks+=1

#         (pdgID,recoverP4,charge,pandora_index_pfo)=recoverTrack(track,pfobjects)

#         P4Recover[countTracks-1]=recoverP4
#         PDGIDRecover[countTracks-1]=pdgID
#         ChargeRecover[countTracks-1]=charge
#         PFIndexRecover[countTracks-1]=pandora_index_pfo

#     return (PDGIDRecover,P4Recover,ChargeRecover,PFIndexRecover)

#!/usr/bin/env python3
"""
analyze_particles.py
====================
Análisis estadístico de hits y energía por tipo de partícula (gen y reco/Pandora)
a partir de archivos ROOT de simulación full-sim (EDM4HEP).

Para cada tipo de partícula (PDG) calcula:
  1. Distribución de la fracción de energía HCAL / ECAL
  2. Distribución del número de hits en cada subdetector (track, ECAL, HCAL, muon)

Uso:
  python analyze_particles.py -i archivo.root
  python analyze_particles.py -i directorio/ --all -o resultados/
  
Requiere: podio, ROOT, numpy, pandas, matplotlib
"""
# ...existing code...
import os
# combinations
from itertools import combinations
from modules.plotting import plot_energy_by_subdet, plot_energy_ratio, plot_ecal_hcal_ratio, plot_hit_distributions, plot_hits_ecal_vs_hcal
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import argparse
import os
import sys
import glob
import math
from collections import defaultdict

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import time
# Importar podio para leer archivos ROOT EDM4HEP
try:
    from podio import root_io
except ImportError:
    print("Error: podio no está instalado. Ejecuta esto en un entorno con Key4HEP.")
    sys.exit(1)

# Importar ROOT y función dRAngle
try:
    import ROOT
except ImportError:
    print("Error: ROOT no está disponible.")
    sys.exit(1)

# Importar dRAngle de myutils
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from modules.myutils import dRAngle
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
def get_track_momentum(trackstate, isclic=True):
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
# ──────────────────────────────────────────────────────────────────────────────
# Constantes (siguiendo convert_to_nn_format.py)
# ──────────────────────────────────────────────────────────────────────────────

# Tipos de detector (mismo esquema que convert_to_nn_format.py)
DETECTOR_TYPES = {
    'INNER_TRACKER': 0,
    'ECAL': 1,
    'HCAL': 2,
    'MUON_TRACKER': 3
}
DETECTOR_ID_TO_TYPE = {
    0: 'INNER_TRACKER',
    1: 'ECAL',  
    2: 'HCAL',
    3: 'MUON_TRACKER'
}

# Estados de generador válidos (solo partículas estables finales)
VALID_GEN_STATUS = {1}

# PDG de neutrinos (ignorar en análisis gen)
NEUTRINO_PDGS = {12, 14, 16}

# Códigos PDG → nombre legible (sin distinguir partícula/antipartícula)
PDG_NAMES = {
    11: "epm",
    13: "μpm",
    22: "gamma",
    111: "π0",
    211: "πpm",
    321: "Kpm",
    2212: "p_bar_p",
    2112: "n_bar_n",
    130: "K_L0",
    310: "K_S0",
    15: "τpm",
}

# Colecciones de calorímetro con su tipo de detector
ECAL_COLLECTIONS = [
    ("ECALBarrel", DETECTOR_TYPES['ECAL']),
    ("ECALEndcap", DETECTOR_TYPES['ECAL']),
    ("ECALOther", DETECTOR_TYPES['ECAL']),
]
HCAL_COLLECTIONS = [
    ("HCALBarrel", DETECTOR_TYPES['HCAL']),
    ("HCALEndcap", DETECTOR_TYPES['HCAL']),
    ("HCALOther", DETECTOR_TYPES['HCAL']),
]
MUON_COLLECTIONS = [
    ("MUON", DETECTOR_TYPES['MUON_TRACKER']),
]
TRACK_COLLECTION = "SiTracks_Refitted"

ALL_CALO_COLLECTIONS = ECAL_COLLECTIONS + HCAL_COLLECTIONS + MUON_COLLECTIONS

def pdg_name(pid):
    """Nombre legible para un código PDG."""
    pid_int = int(pid)
    return PDG_NAMES.get(pid_int, str(pid_int))


# ──────────────────────────────────────────────────────────────────────────────
# Funciones de análisis
# ──────────────────────────────────────────────────────────────────────────────

def build_hit_type_map(event):
    """
    Construye un diccionario que mapea (collectionID, index) -> detector_type.
    Esto permite identificar el tipo de cada hit cuando se accede desde PFO clusters.
    """
    hit_type_map = {}  # (collectionID, index) -> detector_type
    hit_energy_map = {}  # (collectionID, index) -> energy
    
    for coll_name, detector_type in ALL_CALO_COLLECTIONS:
        try:
            coll = event.get(coll_name)
            for hit in coll:
                # print(help(hit))
                # exit()
                obj_id = hit.getObjectID()
                key = (obj_id.collectionID, obj_id.index)
                hit_type_map[key] = detector_type
                hit_energy_map[key] = hit.getEnergy()
        except Exception:
            pass
    
    return hit_type_map, hit_energy_map


def analyze_gen_particles(event, df_reco_mc_links, hit_type_map, hit_energy_map, verbose=False):
    """
    Analiza las partículas generadas (MCParticles) con gen_status == 1.
    Usa CalohitMCTruthLink para asociar hits a cada partícula.
    
    Devuelve una lista de diccionarios con estadísticas por partícula.
    """
    
    # print(df_reco_mc_links)
   
        
    # exit(0)

    try:
        calo_truth_links = list(event.get("CalohitMCTruthLink"))
    except Exception:
        calo_truth_links = []
    
    try:
        track_truth_links = list(event.get("SiTracksMCTruthLink"))
    except Exception:
        track_truth_links = []
    
    # Construir lookup: qué colección pertenece cada hit
    # hit_type_map, hit_energy_map = build_hit_type_map(event)
    
    # Mapeo: mc_index -> {n_track, n_ecal, n_hcal, n_muon, E_ecal, E_hcal, E_muon}
    mc_stats = defaultdict(lambda: {
        'n_track': 0, 'n_ecal': 0, 'n_hcal': 0, 'n_muon': 0,
        'E_ecal': 0.0, 'E_hcal': 0.0, 'E_muon': 0.0
    })
    
    # Procesar calo truth links (solo para hits, no para matching)
    for link in calo_truth_links:
        try:
            hit_obj = link.getRec()
            mc_obj = link.getSim()
            mc_idx = mc_obj.getObjectID().index
            
            hit_obj_id = hit_obj.getObjectID()
            key = (hit_obj_id.collectionID, hit_obj_id.index)
            
            if key in hit_type_map:
                detector_type = hit_type_map[key]
                hit_energy = hit_energy_map.get(key, 0.0)
                
                if detector_type == DETECTOR_TYPES['ECAL']:
                    mc_stats[mc_idx]['n_ecal'] += 1
                    mc_stats[mc_idx]['E_ecal'] += hit_energy
                elif detector_type == DETECTOR_TYPES['HCAL']:
                    mc_stats[mc_idx]['n_hcal'] += 1
                    mc_stats[mc_idx]['E_hcal'] += hit_energy
                elif detector_type == DETECTOR_TYPES['MUON_TRACKER']:
                    mc_stats[mc_idx]['n_muon'] += 1
                    mc_stats[mc_idx]['E_muon'] += hit_energy
        except Exception:
            pass
    
    # Procesar track truth links
    for link in track_truth_links:
        try:
            mc_obj = link.getSim()
            mc_idx = mc_obj.getObjectID().index
            mc_stats[mc_idx]['n_track'] += 1
        except Exception:
            pass
    
    try:
        mc_particles = event.get("MCParticles")
    except Exception:
        if verbose:
            print("    No se encontró colección MCParticles")
        return []   
    # Extraer estadísticas por partícula (solo gen_status en VALID_GEN_STATUS)
    particles = []
    for part in mc_particles:
        try:
            # Usar getObjectID().index para consistencia con mc_stats y df_reco_mc_links
            idx = part.getObjectID().index
            
            gen_status = part.getGeneratorStatus()
            if gen_status not in VALID_GEN_STATUS:
                continue
            
            pid_raw = part.getPDG()
            # Ignorar neutrinos
            if abs(pid_raw) in NEUTRINO_PDGS:
                continue
            
            momentum = part.getMomentum()
            p = math.sqrt(momentum.x**2 + momentum.y**2 + momentum.z**2)
            if p < 1e-10:
                continue  # Saltar partículas sin momento
            
            # Usar abs(pid) para no distinguir partícula/antipartícula
            pid = abs(pid_raw)
            energy = part.getEnergy()
            
            # Calcular ángulos
            theta = math.acos(momentum.z / p) if p > 0 else 0
            phi = math.atan2(momentum.y, momentum.x)
            
            # Obtener estadísticas de hits
            stats = mc_stats.get(idx, {
                'n_track': 0, 'n_ecal': 0, 'n_hcal': 0, 'n_muon': 0,
                'E_ecal': 0.0, 'E_hcal': 0.0, 'E_muon': 0.0
            })
            
            # Filtrar partículas sin ninguna señal en el detector
            n_total_signals = stats['n_track'] + stats['n_ecal'] + stats['n_hcal'] + stats['n_muon']
            if n_total_signals == 0:
                continue  # No interacciona con el detector
            
            # Obtener información de matching desde df_reco_mc_links
            match_rows = df_reco_mc_links[df_reco_mc_links["gen"] == idx]
            if not match_rows.empty:
                matched_reco_idx = match_rows.iloc[0]["reco"]
                is_matched = matched_reco_idx != -999
            else:
                matched_reco_idx = -999
                is_matched = False
            
            # Calcular ratio
            E_ecal = stats['E_ecal']
            E_hcal = stats['E_hcal']
            ratio_hcal_ecal = E_hcal / E_ecal if E_ecal > 0 else np.nan
            
            particles.append({
                "mc_idx": idx,
                "pid": pid,
                "pid_name": pdg_name(pid),
                "E_gen": energy,
                "p": p,
                "theta": theta,
                "phi": phi,
                "gen_status": gen_status,
                "n_track": stats['n_track'],
                "n_ecal": stats['n_ecal'],
                "n_hcal": stats['n_hcal'],
                "n_muon": stats['n_muon'],
                "n_total": stats['n_track'] + stats['n_ecal'] + stats['n_hcal'] + stats['n_muon'],
                "E_ecal": E_ecal,
                "E_hcal": E_hcal,
                "E_muon": stats['E_muon'],
                "ratio_hcal_ecal": ratio_hcal_ecal,
                "is_matched": is_matched,
                "matched_reco_idx": matched_reco_idx
            })
        except Exception as e:
            if verbose:
                print(f"    Error procesando partícula gen {idx}: {e}")
    
    return particles


def analyze_pfos(event,df_reco_mc_links, hit_type_map, hit_energy_map, verbose=False):
    """
    Analiza los PandoraPFOs (partículas reconstruidas).
    
    Para cada PFO cuenta los hits en ECAL, HCAL, tracks y muon usando el mapeo previo.
    """
    try:
        pfos = event.get("PandoraPFOs")
    except Exception:
        if verbose:
            print("    No se encontró colección PandoraPFOs")
        return []
    
    particles = []
    # for pfo in pfos:
    #     # Get track
    #     pfo_track = pfo.getTracks()
    #     n_tracks = len(pfo_track) if pfo_track else 0
    #     # if n_tracks>0:
    #         # print(help(pfo_track[0]))
    #         # exit(0)
    #     for i in range(n_tracks):
    #         print("Associated track", pfo_track[i].getObjectID().index)
    #         print("Chi2", pfo_track[i].getChi2())
    #         # print("Momentum", pfo_track[i].getMomentum().x, pfo_track[i].getMomentum().y, pfo_track[i].getMomentum().z)
    # sitracks = event.get("SiTracks_Refitted")
    # print("SiTracks_Refitted:")
    # for i in range(len(sitracks)):
    #     print("SiTrack", sitracks[i].getObjectID().index)
    #     print("Chi2", sitracks[i].getChi2())
    #     # print("Momentum", sitracks[i].getMomentum().x, sitracks[i].getMomentum().y, sitracks[i].getMomentum().z)
        
        
    # exit(0)
    for pfo_idx, pfo in enumerate(pfos):
        try:
            # Información básica del PFO
            momentum = pfo.getMomentum()
            px, py, pz = momentum.x, momentum.y, momentum.z
            p = math.sqrt(px**2 + py**2 + pz**2)
            energy = pfo.getEnergy()
            pid = pfo.getPDG()
            charge = pfo.getCharge()
            
            theta = math.acos(pz / p) if p > 0 else 0
            phi = math.atan2(py, px)
            
            # Usar abs(pid) para no distinguir partícula/antipartícula
            pid = abs(pid)
            
            # Contar tracks asociados
            tracks = pfo.getTracks()
            n_tracks = len(tracks) if tracks else 0
            
            # Contar hits de calorímetro por tipo usando el mapeo
            clusters = pfo.getClusters()
            if len(clusters) > 1:
                print("nº de clúster pfo", len(clusters))
            n_ecal = 0
            n_hcal = 0
            n_muon = 0
            E_ecal = 0.0
            E_hcal = 0.0
            E_muon = 0.0
            
            for cluster in clusters:
                cluster_hits = cluster.getHits()
                
                for hit in cluster_hits:
                    hit_obj_id = hit.getObjectID()
                    key = (hit_obj_id.collectionID, hit_obj_id.index)
                    
                    # Buscar el tipo de detector en el mapeo
                    if key in hit_type_map:
                        detector_type = hit_type_map[key]
                        hit_energy = hit_energy_map.get(key, hit.getEnergy())
                        
                        if detector_type == DETECTOR_TYPES['ECAL']:
                            n_ecal += 1
                            E_ecal += hit_energy
                        elif detector_type == DETECTOR_TYPES['HCAL']:
                            n_hcal += 1
                            E_hcal += hit_energy
                        elif detector_type == DETECTOR_TYPES['MUON_TRACKER']:
                            n_muon += 1
                            E_muon += hit_energy
            
            # Calcular ratio
            ratio_hcal_ecal = E_hcal / E_ecal if E_ecal > 0 else np.nan
            
            # Obtener información de matching desde df_reco_mc_links
            match_rows = df_reco_mc_links[df_reco_mc_links["reco"] == pfo_idx]
            if not match_rows.empty:
                matched_gen_idx = match_rows.iloc[0]["gen"]
                is_matched = matched_gen_idx != -999
            else:
                matched_gen_idx = -999
                is_matched = False
            
            particles.append({
                "pfo_id": pfo_idx,
                "pid": pid,
                "pid_name": pdg_name(pid),
                "charge": charge,
                "E_reco": energy,
                "p": p,
                "theta": theta,
                "phi": phi,
                "n_track": n_tracks,
                "n_ecal": n_ecal,
                "n_hcal": n_hcal,
                "n_muon": n_muon,
                "n_total": n_tracks + n_ecal + n_hcal + n_muon,
                "E_ecal": E_ecal,
                "E_hcal": E_hcal,
                "E_muon": E_muon,
                "ratio_hcal_ecal": ratio_hcal_ecal,
                "is_matched": is_matched,
                "matched_gen_idx": matched_gen_idx
            })
            
        except Exception as e:
            if verbose:
                print(f"    Error procesando PFO {pfo_idx}: {e}")
    # exit(0)
    return particles

def get_reco_mc_links(event, hit_type_map, hit_energy_map, verbose=False):
    try:
        mc_particles = event.get("MCParticles")
    except Exception:
        return []
    try:
        pfos = event.get("PandoraPFOs")
    except Exception:
        if verbose:
            print("    No se encontró colección PandoraPFOs")
    
    # Construir mc_stats para filtrar partículas sin señal
    try:
        calo_truth_links = list(event.get("CalohitMCTruthLink"))
    except Exception:
        calo_truth_links = []
    
    try:
        track_truth_links = list(event.get("SiTracksMCTruthLink"))
    except Exception:
        track_truth_links = []
    
    mc_stats = defaultdict(lambda: {'n_track': 0, 'n_ecal': 0, 'n_hcal': 0, 'n_muon': 0})
    
    for link in calo_truth_links:
        try:
            hit_obj = link.getRec()
            mc_obj = link.getSim()
            mc_idx = mc_obj.getObjectID().index
            hit_obj_id = hit_obj.getObjectID()
            key = (hit_obj_id.collectionID, hit_obj_id.index)
            if key in hit_type_map:
                detector_type = hit_type_map[key]
                if detector_type == DETECTOR_TYPES['ECAL']:
                    mc_stats[mc_idx]['n_ecal'] += 1
                elif detector_type == DETECTOR_TYPES['HCAL']:
                    mc_stats[mc_idx]['n_hcal'] += 1
                elif detector_type == DETECTOR_TYPES['MUON_TRACKER']:
                    mc_stats[mc_idx]['n_muon'] += 1
        except Exception:
            pass
    
    for link in track_truth_links:
        try:
            mc_obj = link.getSim()
            mc_idx = mc_obj.getObjectID().index
            mc_stats[mc_idx]['n_track'] += 1
        except Exception:
            pass
    
    # Obtener truth links para calo y tracks
    try:
        reco_mc_truth_links = list(event.get("RecoMCTruthLink"))
    except Exception:
        reco_mc_truth_links = []
    # print(help(type(reco_mc_truth_links[0])))
    reco_gen_link = {"gen":[], "reco":[], "weight":[], "Gen_pid":[], "Reco_pid":[]}
    for link in reco_mc_truth_links:
        gen_idx = link.getSim().getObjectID().index
        # Filtrar partículas sin señal en el detector
        stats = mc_stats.get(gen_idx, {'n_track': 0, 'n_ecal': 0, 'n_hcal': 0, 'n_muon': 0})
        n_total = stats['n_track'] + stats['n_ecal'] + stats['n_hcal'] + stats['n_muon']
        if n_total == 0:
            continue  # No interacciona con el detector
        
        reco_gen_link["gen"].append(gen_idx)
        reco_gen_link["reco"].append(link.getRec().getObjectID().index)
        reco_gen_link["weight"].append(link.getWeight())
        reco_gen_link["Gen_pid"].append(link.getSim().getPDG())
        reco_gen_link["Reco_pid"].append(link.getRec().getPDG())
    df_reco_mc_links = pd.DataFrame(reco_gen_link)
    df_reco_mc_links = df_reco_mc_links.sort_values("gen").reset_index(drop=True)
    
    # select only the rows that have the max weight for each gen particle
    df_reco_mc_links = df_reco_mc_links.loc[df_reco_mc_links.groupby("gen")["weight"].idxmax()].reset_index(drop=True)
    # print(df_temp)
    df_reco_mc_links.drop(columns=["weight"], inplace=True)
    
    # Add gen particles with status 1 that are not in the reco_gen_link (solo si tienen señal)
    gen_indices = set(df_reco_mc_links["gen"])
    new_rows = []
    for part in mc_particles:
        try:
            gen_status = part.getGeneratorStatus()
            if gen_status not in VALID_GEN_STATUS:
                continue
            idx = part.getObjectID().index
            if idx not in gen_indices:
                # Filtrar partículas sin señal en el detector
                stats = mc_stats.get(idx, {'n_track': 0, 'n_ecal': 0, 'n_hcal': 0, 'n_muon': 0})
                n_total = stats['n_track'] + stats['n_ecal'] + stats['n_hcal'] + stats['n_muon']
                if n_total == 0:
                    continue  # No interacciona con el detector
                new_rows.append({"gen": idx, "reco": -999, "Gen_pid": part.getPDG(), "Reco_pid": -999})
        except Exception:
            pass
    
    reco_indices = set(df_reco_mc_links["reco"])
    for pfo in pfos:
        reco_idx = pfo.getObjectID().index
        if reco_idx not in reco_indices:
            new_rows.append({"gen": -999, "reco": reco_idx, "Gen_pid": -999, "Reco_pid": -999})
    
    if new_rows:
        df_reco_mc_links = pd.concat([df_reco_mc_links, pd.DataFrame(new_rows)], ignore_index=True)
    # print(df_reco_mc_links)
    return df_reco_mc_links


def get_reco_mc_links_by_dR(event, hit_type_map, hit_energy_map, verbose=False):
    """
    Asocia partículas gen con partículas reco usando distancia dR en el plano (theta, phi).
    
    Para cada partícula gen (status 1, sin neutrinos, con señal en detector):
    - Encuentra la partícula reco más cercana usando dRAngle(p1, p2)
    - Ignora la carga
    
    Devuelve un DataFrame similar a get_reco_mc_links pero con asociaciones basadas en dR.
    """
    try:
        mc_particles = event.get("MCParticles")
    except Exception:
        return pd.DataFrame()
    
    try:
        pfos = event.get("PandoraPFOs")
    except Exception:
        if verbose:
            print("    No se encontró colección PandoraPFOs")
        return pd.DataFrame()
    
    # Construir mc_stats para filtrar partículas sin señal
    try:
        calo_truth_links = list(event.get("CalohitMCTruthLink"))
    except Exception:
        calo_truth_links = []
    
    try:
        track_truth_links = list(event.get("SiTracksMCTruthLink"))
    except Exception:
        track_truth_links = []
    
    mc_stats = defaultdict(lambda: {'n_track': 0, 'n_ecal': 0, 'n_hcal': 0, 'n_muon': 0})
    
    for link in calo_truth_links:
        try:
            hit_obj = link.getRec()
            mc_obj = link.getSim()
            mc_idx = mc_obj.getObjectID().index
            hit_obj_id = hit_obj.getObjectID()
            key = (hit_obj_id.collectionID, hit_obj_id.index)
            if key in hit_type_map:
                detector_type = hit_type_map[key]
                if detector_type == DETECTOR_TYPES['ECAL']:
                    mc_stats[mc_idx]['n_ecal'] += 1
                elif detector_type == DETECTOR_TYPES['HCAL']:
                    mc_stats[mc_idx]['n_hcal'] += 1
                elif detector_type == DETECTOR_TYPES['MUON_TRACKER']:
                    mc_stats[mc_idx]['n_muon'] += 1
        except Exception:
            pass
    
    for link in track_truth_links:
        try:
            mc_obj = link.getSim()
            mc_idx = mc_obj.getObjectID().index
            mc_stats[mc_idx]['n_track'] += 1
        except Exception:
            pass
    
    # Construir lista de partículas gen válidas (status 1, sin neutrinos, con señal)
    valid_gen_particles = []
    for part in mc_particles:
        try:
            gen_status = part.getGeneratorStatus()
            if gen_status not in VALID_GEN_STATUS:
                continue
            
            pid_raw = part.getPDG()
            if abs(pid_raw) in NEUTRINO_PDGS:
                continue
            
            momentum = part.getMomentum()
            p = math.sqrt(momentum.x**2 + momentum.y**2 + momentum.z**2)
            if p < 1e-10:
                continue
            
            idx = part.getObjectID().index
            
            # Filtrar partículas sin señal en el detector
            stats = mc_stats.get(idx, {'n_track': 0, 'n_ecal': 0, 'n_hcal': 0, 'n_muon': 0})
            n_total = stats['n_track'] + stats['n_ecal'] + stats['n_hcal'] + stats['n_muon']
            if n_total == 0:
                continue  # No interacciona con el detector
            
            valid_gen_particles.append((idx, part, momentum, pid_raw))
        except Exception:
            pass
    
    # Construir lista de PFOs
    valid_pfos = []
    for pfo in pfos:
        try:
            momentum = pfo.getMomentum()
            px, py, pz = momentum.x, momentum.y, momentum.z
            p = math.sqrt(px**2 + py**2 + pz**2)
            if p < 1e-10:
                continue
            
            pfo_idx = pfo.getObjectID().index
            pid = pfo.getPDG()
            energy = pfo.getEnergy()
            valid_pfos.append((pfo_idx, momentum, pid, energy))
        except Exception:
            pass
    
    # Para cada partícula gen, encontrar la reco más cercana
    reco_gen_link = {"gen": [], "reco": [], "dR": [], "Gen_pid": [], "Reco_pid": [], "Gen_energy": [], "Reco_energy": []}
    
    for gen_idx, gen_part, gen_momentum, gen_pid in valid_gen_particles:
        # Crear TLorentzVector para gen
        gen_p4 = ROOT.TLorentzVector()
        gen_p4.SetXYZT(gen_momentum.x, gen_momentum.y, gen_momentum.z, gen_part.getEnergy())
        
        min_dR = 0.1
        best_reco_idx = -999
        best_reco_pid = -999
        best_reco_p4 = None
        # Buscar el reco más cercano
        for reco_idx, reco_momentum, reco_pid, reco_energy in valid_pfos:
            # Crear TLorentzVector para reco
            # Nota: no tenemos energía directa del momentum, así que usamos E = sqrt(p^2 + m^2)
            # Para simplificar, asumimos masa 0 (fotón) o calculamos E ~ p para relativistas
            # p_reco = math.sqrt(reco_momentum.x**2 + reco_momentum.y**2 + reco_momentum.z**2)
            reco_p4 = ROOT.TLorentzVector()
            reco_p4.SetXYZM(reco_momentum.x, reco_momentum.y, reco_momentum.z, 0.0)
            
            # Calcular dR
            dR = dRAngle(gen_p4, reco_p4)
            
            if dR < min_dR:
                min_dR = dR
                best_reco_idx = reco_idx
                best_reco_pid = reco_pid
                best_reco_p4 = reco_p4
        
        # Agregar asociación
        reco_gen_link["gen"].append(gen_idx)
        reco_gen_link["reco"].append(best_reco_idx)
        reco_gen_link["dR"].append(min_dR)
        reco_gen_link["Gen_pid"].append(gen_pid)
        reco_gen_link["Reco_pid"].append(best_reco_pid)

        reco_gen_link["Gen_energy"].append(gen_p4.P())


        reco_gen_link["Reco_energy"].append(best_reco_p4.P() if best_reco_p4 else np.nan)

    
    df_reco_mc_links = pd.DataFrame(reco_gen_link)
    # print(df_reco_mc_links)
    if df_reco_mc_links.empty:
        return df_reco_mc_links
    
    df_reco_mc_links = df_reco_mc_links.sort_values("gen").reset_index(drop=True)
    
    # Para cada gen, mantener solo el match con mínimo dR
    df_reco_mc_links = df_reco_mc_links.loc[df_reco_mc_links.groupby("gen")["dR"].idxmin()].reset_index(drop=True)
    
    # Añadir gen sin match (los que no están en la lista)
    gen_indices = set(df_reco_mc_links["gen"])
    new_rows = []
    for gen_idx, gen_part, gen_momentum, gen_pid in valid_gen_particles:
        if gen_idx not in gen_indices:
            new_rows.append({"gen": gen_idx, "reco": -999, "dR": np.nan, "Gen_pid": gen_pid, "Reco_pid": -999, "Gen_energy": gen_part.getEnergy()})
    
    # Añadir reco sin match (los que no están en la lista)
    reco_indices = set(df_reco_mc_links["reco"])
    for reco_idx, reco_momentum, reco_pid, reco_energy in valid_pfos:
        if reco_idx not in reco_indices:
            new_rows.append({"gen": -999, "reco": reco_idx, "dR": np.nan, "Gen_pid": -999, "Reco_pid": reco_pid, "Reco_energy": reco_energy})
    
    if new_rows:
        df_reco_mc_links = pd.concat([df_reco_mc_links, pd.DataFrame(new_rows)], ignore_index=True)
    
    # Eliminar columna dR para mantener compatibilidad con el resto del código
    df_reco_mc_links = df_reco_mc_links.drop(columns=["dR"])
    
    return df_reco_mc_links


def process_root_file_dR(filepath, max_events=None, verbose=False):
    """
    Procesa un archivo ROOT y devuelve DataFrames de partículas gen y reco.
    Usa asociaciones por distancia dR en lugar de RecoMCTruthLink.
    """
    if verbose:
        print(f"  Abriendo {filepath} (dR)...")
    
    try:
        reader = root_io.Reader(filepath)
    except Exception as e:
        print(f"  Error abriendo {filepath}: {e}")
        return pd.DataFrame(), pd.DataFrame()
    
    all_gen = []
    all_reco = []
    
    
    total_gen_pion = 0
    total_reco_pion_matched_wo_neutron = 0
    total_reco_pion_neutron = 0
    events = reader.get("events")
    n_events = 0
    time_start = time.time()
    neutral_recover = {"event_idx":[], "gen_idx": [], "reco_idx": [], "Track_idx": [], "gen_pid": [], "gen_energy": [], "reco_energy":[], "res_p": [], "res_theta": [], "res_phi": []}
    for event_idx, event in enumerate(events):

        if max_events is not None and event_idx >= max_events:
            break
        if verbose and event_idx % 100 == 0:
            time_end = time.time()
            print(f"    Evento {event_idx}... in {(time_end - time_start)/60:.2f} min (dR)")
            time_start = time.time()
        
        # Construir mapeo de hits una vez por evento
        hit_type_map, hit_energy_map = build_hit_type_map(event)
        
        # Usar asociaciones por dR
        df_reco_mc_links = get_reco_mc_links_by_dR(event, hit_type_map, hit_energy_map, verbose=verbose)
        # Analizar partículas gen
        gen_particles = analyze_gen_particles(event, df_reco_mc_links, hit_type_map, hit_energy_map, verbose)
        
        for p in gen_particles:
            p["event_idx"] = event_idx
        all_gen.extend(gen_particles)
        
        # Analizar PFOs
        reco_particles = analyze_pfos(event, df_reco_mc_links, hit_type_map, hit_energy_map, verbose)
        for p in reco_particles:
            p["event_idx"] = event_idx
        all_reco.extend(reco_particles)
        
        n_events += 1
    
        pfos = event.get("PandoraPFOs")
        genParticles = event.get("MCParticles")
        # print(df_reco_mc_links)
        
        if (df_reco_mc_links["Reco_pid"] == 2112).any(): # neutron case
          # idx where reco_id is 2112
            # idx_reco_neutron = df_reco_mc_links[df_reco_mc_links["Reco_pid"] == 2112].index
            associated_tracks = []
            for pfo in pfos:
                # Get track
                pfo_track = pfo.getTracks()
                n_tracks = len(pfo_track) if pfo_track else 0
                # if n_tracks>0:
                    # print(help(pfo_track[0]))
                    # exit(0)
                for i in range(n_tracks):
                    # print("Associated track", pfo_track[i].getObjectID().index)
                    # print("Chi2", pfo_track[i].getChi2())
                    associated_tracks.append(pfo_track[i].getObjectID().index)
                    # print("Momentum", pfo_track[i].getMomentum().x, pfo_track[i].getMomentum().y, pfo_track[i].getMomentum().z)
            sitracks = event.get("SiTracks_Refitted")
            non_associated_tracks = {"track_idx":[], "p":[], "px":[], "py":[], "pz":[], "theta":[], "phi":[], "energy":[], "charge":[]}
            for i in range(len(sitracks)):
                si_track_idx = sitracks[i].getObjectID().index
                if si_track_idx not in associated_tracks:
                    # print(help(sitracks[i]))
                    trackstate = sitracks[i].getTrackStates()[0]
                    # print(help(trackstate))
                    # exit(0)
                    p, theta, phi, energy, px, py, pz = get_track_momentum(trackstate, isclic=False)
                    non_associated_tracks["track_idx"].append(si_track_idx)
                    non_associated_tracks["p"].append(p)
                    non_associated_tracks["px"].append(px)
                    non_associated_tracks["py"].append(py)
                    non_associated_tracks["pz"].append(pz)
                    non_associated_tracks["theta"].append(theta)
                    non_associated_tracks["phi"].append(phi)
                    non_associated_tracks["energy"].append(energy)
                    non_associated_tracks["charge"].append(-1 if trackstate.omega < 0 else 1)
            non_associated_tracks_df = pd.DataFrame(non_associated_tracks)
            # n unique pfo id with pid 2112
            idx_pfo_neutron = df_reco_mc_links[df_reco_mc_links["Reco_pid"] == 2112]["reco"].unique()
            
            save_extra_data = True
            print("IDX PFO neutron:", idx_pfo_neutron, " Event idx:", event_idx)
            if len(idx_pfo_neutron) == 1 and not non_associated_tracks_df.empty:
                path_to_save = ""
                if not os.path.exists(f"neutral_recover_images/event_{event_idx}/") and save_extra_data:
                    os.makedirs(f"neutral_recover_images/event_{event_idx}/")
                if save_extra_data:    
                    path_to_save = f"neutral_recover_images/event_{event_idx}/"
                    df_reco_mc_links.to_csv(f"{path_to_save}event_{event_idx}_reco_mc_links.csv", index=False)
                
                # One reco neutron case
                process_one_neutron(pfos, idx_pfo_neutron,
                                    hit_type_map, non_associated_tracks_df, event_idx,
                                    df_reco_mc_links, genParticles, sitracks, neutral_recover, path_to_save, save_extra_data)
            elif len(idx_pfo_neutron) > 1 and not non_associated_tracks_df.empty:
                path_to_save = ""
                if not os.path.exists(f"neutral_recover_images/event_{event_idx}/") and save_extra_data:
                    os.makedirs(f"neutral_recover_images/event_{event_idx}/")
                if save_extra_data:
                    path_to_save = f"neutral_recover_images/event_{event_idx}/"
                    df_reco_mc_links.to_csv(f"{path_to_save}event_{event_idx}_reco_mc_links.csv", index=False)
                for idx in idx_pfo_neutron:
                    process_one_neutron(pfos, [idx],
                                        hit_type_map, non_associated_tracks_df, event_idx,
                                        df_reco_mc_links, genParticles, sitracks, neutral_recover, path_to_save, save_extra_data)

            # calc pion efficiency as number of gen pions with reco match (and reco pdg is 211) / number of gen pions with signal in detector
            matched_reco_pion = df_reco_mc_links[(abs(df_reco_mc_links["Gen_pid"]) == 211) & (abs(df_reco_mc_links["Reco_pid"]) == 211)]
            # Check if one reco pion is matched to more than one gen pion
            # print("Matched reco pions (without neutron case):", matched_reco_pion)
            unique_reco_pion_idx = matched_reco_pion["reco"].unique()
            total_matches = len(matched_reco_pion)
            for idx in unique_reco_pion_idx:
                n_matches = len(matched_reco_pion[matched_reco_pion["reco"] == idx])
                if n_matches > 1:
                    # print(f"Reco pion {idx} is matched to {n_matches} gen pions")
                    total_matches -= 1  # restar los matches duplicados
            gen_pions_with_signal = df_reco_mc_links[abs(df_reco_mc_links["Gen_pid"]) == 211]
            # pion_efficiency = total_matches / len(gen_pions_with_signal) if len(gen_pions_with_signal) > 0 else 0
            total_gen_pion += len(gen_pions_with_signal)
            total_reco_pion_matched_wo_neutron += total_matches
            total_reco_pion_neutron += total_matches
            # print(f"Total gen pions with signal: {total_gen_pion}")
            # print(f"Total reco pions matched without neutron case: {total_reco_pion_matched_wo_neutron}")
            # print(f"Total reco pions matched with neutron case: {total_reco_pion_neutron}")
            
    neutral_recover_df = pd.DataFrame(neutral_recover)
    
    file_name = os.path.basename(filepath).split(".")[0]
    
    # add file name to every row of neutral_recover_df
    neutral_recover_df["file_name"] = file_name
    #  obtain the name of the file without extension
    # calc pion efficiency adding reco neutron
    if not neutral_recover_df.empty and neutral_recover_df["gen_pid"].isin([211, -211]).any():
        matched_reco_pion_neutron = neutral_recover_df[neutral_recover_df["gen_pid"].isin([211, -211])]
        unique_reco_pion_neutron_idx = matched_reco_pion_neutron["reco_idx"].unique()
        total_matches_neutron = len(matched_reco_pion_neutron)
        for idx in unique_reco_pion_neutron_idx:
            n_matches = len(matched_reco_pion_neutron[matched_reco_pion_neutron["reco_idx"] == idx])
            if n_matches > 1:
                # print(f"Reco pion/neutron {idx} is matched to {n_matches} gen pions")
                total_matches_neutron -= 1  # restar los matches duplicados
        total_reco_pion_neutron += total_matches_neutron
        
        # Multiple reco neutron case
        # return df_gen, df_reco
            
    # neutral_recover_df.to_csv(f"neutral_recover_images/{file_name}_event_neutral_recover.csv", index=False)
            
            
    df_gen = pd.DataFrame(all_gen) if all_gen else pd.DataFrame()
    df_reco = pd.DataFrame(all_reco) if all_reco else pd.DataFrame()
    
    pion_efficiency = total_reco_pion_matched_wo_neutron / total_gen_pion if total_gen_pion > 0 else 0
    pion_efficiency_neutron = total_reco_pion_neutron / total_gen_pion if total_gen_pion > 0 else 0
    return df_gen, df_reco, total_gen_pion, total_reco_pion_matched_wo_neutron, total_reco_pion_neutron, pion_efficiency, pion_efficiency_neutron, neutral_recover_df

def process_one_neutron(pfos, idx_pfo_neutron, hit_type_map,
                        non_associated_tracks_df, event_idx,
                        df_reco_mc_links,genParticles, sitracks, neutral_recover, path_to_save, save_extra_data=False):
    track_centroid_assoc = {"Pfo_idx": [],
                            "Track_idx": [],
                            "Ang_dist_hcal": [],
                            "Compatible_hcal": [],
                            "Ang_dist_ecal": [],
                            "Compatible_ecal": [],
                            "Ang_dist_total": [],
                            "Compatible_total": []}
    
    # Get pfo with idx_pfo_neutron
    for pfo in pfos:

        pfo_idx = pfo.getObjectID().index
        if pfo_idx == idx_pfo_neutron[0]:
            break
        
    clusters = pfo.getClusters()
    for cluster in clusters:
        cluster_hits = cluster.getHits()
        hits_hcal = []
        hits_ecal = []
        for hit in cluster_hits:
            hit_obj_id = hit.getObjectID()
            key = (hit_obj_id.collectionID, hit_obj_id.index)
            if key in hit_type_map:
                detector_type = hit_type_map[key]
                hit_x = hit.getPosition().x
                hit_y = hit.getPosition().y
                hit_z = hit.getPosition().z
                if detector_type == DETECTOR_TYPES['ECAL']:
                    hits_ecal.append([hit_x, hit_y, hit_z])
                elif detector_type == DETECTOR_TYPES['HCAL']:
                    hits_hcal.append([hit_x, hit_y, hit_z])

    print("N hits HCAL:", len(hits_hcal), "N hits ECAL:", len(hits_ecal))
    print("Mean position HCAL hits:", np.mean(hits_hcal, axis=0) if hits_hcal else "N/A")

    if hits_hcal:
        hits_hcal = np.array(hits_hcal)


    if hits_ecal:
        hits_ecal = np.array(hits_ecal)

    for track_row in non_associated_tracks_df.itertuples():
        si_track_idx = track_row.track_idx
        hcal_signal = len(hits_hcal) > 0
        ecal_signal = len(hits_ecal) > 0
        if ecal_signal and hcal_signal:
            total_hits = np.concatenate((hits_hcal, hits_ecal), axis=0)
            total_centroid = np.mean(total_hits, axis=0)
            ecal_centroid = np.mean(hits_ecal, axis=0)
            hcal_centroid = np.mean(hits_hcal, axis=0)
        elif hcal_signal:
            ecal_centroid = np.array([0, 0, 0])
            hcal_centroid = np.mean(hits_hcal, axis=0)
            total_centroid = np.mean(hits_hcal, axis=0)
        elif ecal_signal:
            hcal_centroid = np.array([0, 0, 0])
            ecal_centroid = np.mean(hits_ecal, axis=0)
            total_centroid = np.mean(hits_ecal, axis=0)
        
        
        track_p4 = ROOT.TLorentzVector()
        px = track_row.px
        py = track_row.py
        pz = track_row.pz
        track_p4.SetXYZM(px, py, pz, 0.0)
        track_centroid_assoc["Pfo_idx"].append(pfo.getObjectID().index)
        track_centroid_assoc["Track_idx"].append(si_track_idx)
        total_p4 = ROOT.TLorentzVector()
        total_p4.SetXYZM(total_centroid[0], total_centroid[1], total_centroid[2], 0.0)
        dR_total = dRAngle(track_p4, total_p4)                    
        track_centroid_assoc["Compatible_total"].append(dR_total < 0.1)
        track_centroid_assoc["Ang_dist_total"].append(dR_total)
        
        if hcal_signal:
            hcal_p4 = ROOT.TLorentzVector()
            hcal_p4.SetXYZM(hcal_centroid[0], hcal_centroid[1], hcal_centroid[2], 0.0)
            dR = dRAngle(track_p4, hcal_p4)
            track_centroid_assoc["Ang_dist_hcal"].append(dR)
            track_centroid_assoc["Compatible_hcal"].append(dR < 0.1)
        else:
            track_centroid_assoc["Ang_dist_hcal"].append(np.nan)
            track_centroid_assoc["Compatible_hcal"].append(False)
            
        if ecal_signal:
            ecal_p4 = ROOT.TLorentzVector()
            ecal_p4.SetXYZM(ecal_centroid[0], ecal_centroid[1], ecal_centroid[2], 0.0)
            dR_ecal = dRAngle(track_p4, ecal_p4)
            track_centroid_assoc["Ang_dist_ecal"].append(dR_ecal)
            track_centroid_assoc["Compatible_ecal"].append(dR_ecal < 0.1)
        else:
            track_centroid_assoc["Ang_dist_ecal"].append(np.nan)
            track_centroid_assoc["Compatible_ecal"].append(False)

                
    trk_assoc_df = pd.DataFrame(track_centroid_assoc)
    if save_extra_data:
        trk_assoc_df.to_csv(f"{path_to_save}event_{event_idx}_pfo_{pfo.getObjectID().index}_track_centroid_assoc.csv", index=False)
    if trk_assoc_df["Compatible_total"].any():
        # Check that only one track is compatible
        compatible_tracks = trk_assoc_df[trk_assoc_df["Compatible_total"]]
        if len(compatible_tracks) == 1:
            # Obtain Track momentum
            print("Compatible track found for PFO", pfo.getObjectID().index)
            track_idx = compatible_tracks.iloc[0]["Track_idx"]
            # Get track info from unassociated tracks dataframe
            track_info = non_associated_tracks_df[non_associated_tracks_df["track_idx"] == track_idx].iloc[0]
            p = track_info["p"]
            px = track_info["px"]
            py = track_info["py"]
            pz = track_info["pz"]
            energy = track_info["energy"]
            theta = track_info["theta"]
            phi = track_info["phi"]
            charge = track_info["charge"]
            # Look for the pion associated to the neutron in the gen particles
            pfo_idx = pfo.getObjectID().index
            
            # Get Generator Particle info
            gen_idx = df_reco_mc_links[df_reco_mc_links["reco"] == pfo_idx]["gen"].values
            if len(gen_idx) > 0:
                # Check if any is 211
                gen_pids = df_reco_mc_links[df_reco_mc_links["reco"] == pfo_idx]["Gen_pid"].values
                if 211 in gen_pids:
                    # Find the index of the pion (211)
                    pion_idx = np.where(gen_pids == 211)[0][0]
                    gen_idx = gen_idx[pion_idx]
                else:
                    # If no pion is found, set gen_idx to -1 or handle appropriately
                    gen_idx = gen_idx[0]
            gen_pid = df_reco_mc_links[df_reco_mc_links["reco"] == pfo_idx]["Gen_pid"].values[0]
            for gen_part in genParticles:
                if gen_part.getObjectID().index == gen_idx:
                    break
                    
            gen_momentum = gen_part.getMomentum()
            gen_px, gen_py, gen_pz = gen_momentum.x, gen_momentum.y, gen_momentum.z
            mass = gen_part.getMass()
            gen_p4 = ROOT.TLorentzVector()
            gen_p4.SetXYZM(gen_px, gen_py, gen_pz, mass)
            # Compare gen pion momentum with track momentum
            track_p4 = ROOT.TLorentzVector()
            track_p4.SetPxPyPzE(px, py, pz, energy)
            
            # Compare resolution in P
            res_p = (track_p4.P() - gen_p4.P()) / gen_p4.P()
            # Compare resolution in theta
            res_theta = (track_p4.Theta() - gen_p4.Theta()) / gen_p4.Theta()
            # Compare resolution in phi
            res_phi = (track_p4.Phi() - gen_p4.Phi()) / gen_p4.Phi()
            neutral_recover["event_idx"].append(event_idx)
            neutral_recover["gen_idx"].append(gen_idx)
            neutral_recover["reco_idx"].append(pfo_idx)
            neutral_recover["Track_idx"].append(track_idx)
            neutral_recover["gen_pid"].append(gen_pid)
            neutral_recover["gen_energy"].append(gen_part.getEnergy())
            neutral_recover["reco_energy"].append(pfo.getEnergy())
            neutral_recover["res_p"].append(res_p)
            neutral_recover["res_theta"].append(res_theta)
            neutral_recover["res_phi"].append(res_phi)
            associated_track_info = {
                "event_idx": event_idx,
                "pfo_idx": pfo_idx,
                "track_idx": track_idx,
                "gen_idx": gen_idx,
                "gen_pid": gen_pid,
                "track_charge": charge,
                "track_p": track_p4.P(),
                "track_theta": track_p4.Theta(),
                "track_phi": track_p4.Phi(),
                "track_energy": track_p4.E(),
                "gen_p": gen_p4.P(),
                "gen_theta": gen_p4.Theta(),
                "gen_phi": gen_p4.Phi(),
                "gen_energy": gen_p4.E(),
                "res_p": res_p,
                "res_theta": res_theta,
                "res_phi": res_phi}
            assoc_df = pd.DataFrame([associated_track_info])
            if save_extra_data:
                assoc_df.to_csv(f"{path_to_save}event_{event_idx}_pfo_{pfo.getObjectID().index}_track_assoc_detailed.csv", index=False)

        elif len(compatible_tracks) > 1:
            print(f"Multiple compatible tracks found for PFO {pfo.getObjectID().index}: {compatible_tracks} event idx: {event_idx}")
            # Obtain energy of the pfo
            # Try to combine two tracks, sum energy and compare with the pfo energy
            tracks_energy = []
            for track_idx in compatible_tracks["Track_idx"]:
                trackstate = sitracks[int(track_idx)].getTrackStates()[0]
                p, theta, phi, energy, px, py, pz = get_track_momentum(trackstate, isclic=False)
                tracks_energy.append(energy)
            # Try 2 by 2 combinations
            track_info_df = pd.DataFrame({
                "Track_idx": compatible_tracks["Track_idx"].values,
                "Energy": tracks_energy
            })
            if save_extra_data:
                track_info_df.to_csv(f"{path_to_save}event_{event_idx}_pfo_{pfo.getObjectID().index}_compatible_tracks_energy.csv", index=False)
            best_combination = None
            best_diff = float("inf")
            combs_results = {"Combination": [], "Combined_energy": [], "Energy_diff": []}
            for comb in combinations(range(len(tracks_energy)), 2):
                comb_energy = sum(tracks_energy[i] for i in comb)
                diff = abs(comb_energy - pfo.getEnergy())
                combs_results["Combination"].append(comb)
                combs_results["Combined_energy"].append(comb_energy)
                combs_results["Energy_diff"].append(diff)
                if diff < best_diff:
                    best_diff = diff
                    best_combination = comb
            if best_combination is not None:
                print(f"Best combination for PFO {pfo.getObjectID().index}: {best_combination}, energy difference: {best_diff}")
            else:
                print(f"No valid combination found for PFO {pfo.getObjectID().index}")

            # gen assocs
            associated_track_info = {
                "event_idx": [event_idx]*2,
                "pfo_idx": [pfo.getObjectID().index]*2,
                "track_idxs": best_combination,
                "track_energies": [tracks_energy[i] for i in best_combination],
                "pfo_energy": [pfo.getEnergy()]*2,}
            assoc_df = pd.DataFrame(associated_track_info)
            combs_results_df = pd.DataFrame(combs_results)
            if save_extra_data:
                assoc_df.to_csv(f"{path_to_save}event_{event_idx}_pfo_{pfo.getObjectID().index}_track_assoc_combination_detailed.csv", index=False)
                combs_results_df.to_csv(f"{path_to_save}event_{event_idx}_pfo_{pfo.getObjectID().index}_track_assoc_combination_results.csv", index=False)
    else:
        print(f"No compatible tracks found for PFO {pfo.getObjectID().index}")

def print_summary_table(df, level_label):
    """Print a compact summary table grouped by PID."""
    if df.empty:
        print(f"\n  [{level_label}] No data.\n")
        return

    # Numeric columns to aggregate
    hit_cols = [c for c in df.columns if c.startswith("n_")]
    energy_cols = [c for c in df.columns if c.startswith("E_") or c == "ratio_hcal_ecal"]
    agg_cols = {c: ["mean", "std", "median"] for c in hit_cols + energy_cols if c in df.columns}
    agg_cols["pid"] = "count"

    summary = df.groupby("pid_name").agg(agg_cols)
    summary.columns = ["_".join(col).strip("_") for col in summary.columns]
    summary = summary.rename(columns={"pid_count": "N_particles"})
    summary = summary.sort_values("N_particles", ascending=False)

    print(f"\n{'=' * 100}")
    print(f"  Resumen — {level_label}")
    print(f"{'=' * 100}")
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 200)
    pd.set_option("display.float_format", lambda x: f"{x:.3f}")
    print(summary.to_string())
    print()


def get_association_cases(df_gen, df_reco):
    """
    Identifica todos los casos de asociación Gen_pid -> Reco_pid.
    
    Devuelve seis categorías:
    1. exact_matches: diccionario {(gen_pid, reco_pid): count} donde gen_pid == reco_pid
    2. mismatches_individual: diccionario {(gen_pid, reco_pid): count} donde gen_pid != reco_pid (cada pareja única)
    3. mismatches_by_reco: diccionario {reco_pid: [(gen_pid, count), ...]} donde gen_pid != reco_pid, agrupado por reco_pid
    4. mismatches_by_gen: diccionario {gen_pid: [(reco_pid, count), ...]} donde gen_pid != reco_pid, agrupado por gen_pid
    5. unmatched_gen: diccionario {gen_pid: count} para partículas gen sin reco asociado
    6. unmatched_reco: diccionario {reco_pid: count} para partículas reco sin gen asociado
    """
    # Combinar información de ambos DataFrames usando las columnas matched_*_idx
    associations = []
    
    # Extraer asociaciones de partículas gen que tienen match
    if not df_gen.empty and "matched_reco_idx" in df_gen.columns:
        for _, row in df_gen.iterrows():
            if row["is_matched"] and row["matched_reco_idx"] != -999:
                reco_idx = int(row["matched_reco_idx"])
                event_idx = row["event_idx"]
                # Filtrar por event_idx Y pfo_id para evitar confusión entre eventos
                reco_row = df_reco[(df_reco["pfo_id"] == reco_idx) & (df_reco["event_idx"] == event_idx)]
                if not reco_row.empty:
                    gen_pid = abs(int(row["pid"]))
                    reco_pid = abs(int(reco_row.iloc[0]["pid"]))
                    associations.append((gen_pid, reco_pid))
    
    # Contar asociaciones
    from collections import Counter
    assoc_counts = Counter(associations) if associations else {}
    
    # Categorizar matched
    exact_matches = {}
    mismatches_individual = {}
    mismatches_by_reco = defaultdict(list)
    mismatches_by_gen = defaultdict(list)
    
    for (gen_pid, reco_pid), count in assoc_counts.items():
        if gen_pid == reco_pid:
            exact_matches[(gen_pid, reco_pid)] = count
        else:
            mismatches_individual[(gen_pid, reco_pid)] = count
            mismatches_by_reco[reco_pid].append((gen_pid, count))
            mismatches_by_gen[gen_pid].append((reco_pid, count))
    
    # Contar unmatched gen (gen sin reco asociado)
    unmatched_gen = Counter()
    if not df_gen.empty and "is_matched" in df_gen.columns:
        for _, row in df_gen.iterrows():
            if not row["is_matched"] or row.get("matched_reco_idx", -999) == -999:
                gen_pid = abs(int(row["pid"]))
                unmatched_gen[gen_pid] += 1
    
    # Contar unmatched reco (reco sin gen asociado)
    unmatched_reco = Counter()
    if not df_reco.empty and "is_matched" in df_reco.columns:
        for _, row in df_reco.iterrows():
            if not row["is_matched"] or row.get("matched_gen_idx", -999) == -999:
                reco_pid = abs(int(row["pid"]))
                unmatched_reco[reco_pid] += 1
    
    return exact_matches, mismatches_individual, mismatches_by_reco, mismatches_by_gen, dict(unmatched_gen), dict(unmatched_reco)


def filter_by_association(df, is_gen, gen_pid_filter=None, reco_pid_filter=None, match_type="all"):
    """
    Filtra un DataFrame por tipo de asociación Gen-Reco.
    
    Args:
        df: DataFrame a filtrar (gen o reco)
        is_gen: True si df es de partículas gen, False si es reco
        gen_pid_filter: PID de gen a filtrar (None = todos)
        reco_pid_filter: PID de reco a filtrar (None = todos)
        match_type: "exact" (gen_pid == reco_pid), "mismatch" (gen_pid != reco_pid), "all" (todos)
    
    Returns:
        DataFrame filtrado
    """
    if df.empty:
        return df
    
    filtered = df.copy()
    
    if is_gen:
        # Filtrar por partículas gen que tienen match
        filtered = filtered[filtered["is_matched"] == True]
        
        if filtered.empty:
            return filtered
        
        # Obtener información de reco asociado
        # Necesitamos un mapeo de matched_reco_idx -> reco_pid
        # Esto se debe hacer antes de llamar a esta función
        
        # Aplicar filtros de PID
        if gen_pid_filter is not None:
            filtered = filtered[abs(filtered["pid"]) == abs(gen_pid_filter)]
        
        # Filtrar por tipo de match
        if match_type == "exact":
            # Aquí necesitamos comparar gen_pid con reco_pid del match
            # Esto requiere información adicional
            pass
        elif match_type == "mismatch":
            pass
    else:
        # Filtrar por partículas reco que tienen match
        filtered = filtered[filtered["is_matched"] == True]
        
        if filtered.empty:
            return filtered
        
        # Aplicar filtros de PID
        if reco_pid_filter is not None:
            filtered = filtered[abs(filtered["pid"]) == abs(reco_pid_filter)]
    
    return filtered


def generate_association_plots(df_gen, df_reco, output_dir, verbose=False):
    """
    Genera plots para cada caso de asociación Gen_pid -> Reco_pid.
    
    Crea subcarpetas:
    - exact_matches/gen{gen_pid}_reco{reco_pid}/ para aciertos (gen_pid == reco_pid)
    - mismatches_individual/gen{gen_pid}_reco{reco_pid}/ para cada pareja específica donde gen_pid != reco_pid
    - mismatches_by_gen/gen{gen_pid}/ para TODOS los gen_pid que NO se reconstruyen como gen_pid (agrupados)
    - mismatches_by_reco/reco{reco_pid}/ para TODOS los gen que van a reco_pid excepto reco_pid→reco_pid (agrupados)
    """
    if df_gen.empty or df_reco.empty:
        print("\n  No hay datos de gen o reco para generar plots de asociación")
        return
    
    print("\n" + "="*100)
    print("  Generando plots por asociación Gen-Reco")
    print("="*100)
    
    # Identificar casos únicos
    exact_matches, mismatches_individual, mismatches_by_reco, mismatches_by_gen, unmatched_gen, unmatched_reco = get_association_cases(df_gen, df_reco)
    
    print(f"\n  Casos encontrados:")
    print(f"    - Exact matches (aciertos): {len(exact_matches)}")
    print(f"    - Mismatches individuales (parejas únicas): {len(mismatches_individual)}")
    print(f"    - Mismatches agrupados por gen_pid: {len(mismatches_by_gen)}")
    print(f"    - Mismatches agrupados por reco_pid: {len(mismatches_by_reco)}")
    print(f"    - Gen sin reco (unmatched gen): {len(unmatched_gen)} tipos de partícula")
    print(f"    - Reco sin gen (unmatched reco): {len(unmatched_reco)} tipos de partícula")
    
    # 1. Plots para exact matches: gen_pid -> reco_pid (donde gen_pid == reco_pid)
    print("\n  [1/6] Generando plots para exact matches (aciertos)...")
    for (gen_pid, reco_pid), count in exact_matches.items():
        subdir = os.path.join(output_dir, "associations", "exact_matches", 
                              f"gen{gen_pid}_reco{reco_pid}")
        os.makedirs(subdir, exist_ok=True)
        
        # Filtrar gen y reco para este caso específico
        df_gen_filtered = []
        df_reco_filtered = []
        
        for _, gen_row in df_gen.iterrows():
            if not gen_row["is_matched"] or gen_row["matched_reco_idx"] == -999:
                continue
            if abs(int(gen_row["pid"])) != gen_pid:
                continue
            
            # Buscar la partícula reco correspondiente
            reco_matches = df_reco[
                (df_reco["event_idx"] == gen_row["event_idx"]) & 
                (df_reco["pfo_id"] == gen_row["matched_reco_idx"])
            ]
            
            if not reco_matches.empty:
                reco_row = reco_matches.iloc[0]
                if abs(int(reco_row["pid"])) == reco_pid:
                    df_gen_filtered.append(gen_row)
                    df_reco_filtered.append(reco_row)
        
        if not df_gen_filtered:
            continue
        
        df_gen_filtered = pd.DataFrame(df_gen_filtered)
        df_reco_filtered = pd.DataFrame(df_reco_filtered)
        
        label = f"Gen {pdg_name(gen_pid)} → Reco {pdg_name(reco_pid)} (N={count})"
        
        if verbose:
            print(f"    {label}")
        
        # Generar todos los plots para este caso
        plot_energy_ratio(df_gen_filtered, f"{label} - Gen", subdir, pdg_name)
        plot_hit_distributions(df_gen_filtered, f"{label} - Gen", subdir, pdg_name)
        plot_energy_by_subdet(df_gen_filtered, f"{label} - Gen", subdir, pdg_name)
        plot_ecal_hcal_ratio(df_gen_filtered, f"{label} - Gen", subdir, pdg_name)
        plot_hits_ecal_vs_hcal(df_gen_filtered, f"{label} - Gen", subdir, pdg_name)
        
        if not df_reco_filtered.empty:
            plot_energy_ratio(df_reco_filtered, f"{label} - Reco", subdir, pdg_name)
            plot_hit_distributions(df_reco_filtered, f"{label} - Reco", subdir, pdg_name)
            plot_energy_by_subdet(df_reco_filtered, f"{label} - Reco", subdir, pdg_name)
            plot_ecal_hcal_ratio(df_reco_filtered, f"{label} - Reco", subdir, pdg_name)
            plot_hits_ecal_vs_hcal(df_reco_filtered, f"{label} - Reco", subdir, pdg_name)
    
    # 2. Plots para mismatches individuales: cada pareja (gen_pid, reco_pid) donde gen_pid != reco_pid
    print("\n  [2/6] Generando plots para mismatches individuales (cada pareja)...")
    for (gen_pid, reco_pid), count in mismatches_individual.items():
        subdir = os.path.join(output_dir, "associations", "mismatches_individual", 
                              f"gen{gen_pid}_reco{reco_pid}")
        os.makedirs(subdir, exist_ok=True)
        
        # Filtrar gen y reco para esta pareja específica
        df_gen_filtered = []
        df_reco_filtered = []
        
        for _, gen_row in df_gen.iterrows():
            if not gen_row["is_matched"] or gen_row["matched_reco_idx"] == -999:
                continue
            if abs(int(gen_row["pid"])) != gen_pid:
                continue
            
            # Buscar la partícula reco correspondiente
            reco_matches = df_reco[
                (df_reco["event_idx"] == gen_row["event_idx"]) & 
                (df_reco["pfo_id"] == gen_row["matched_reco_idx"])
            ]
            
            if not reco_matches.empty:
                reco_row = reco_matches.iloc[0]
                if abs(int(reco_row["pid"])) == reco_pid:
                    df_gen_filtered.append(gen_row)
                    df_reco_filtered.append(reco_row)
        
        if not df_gen_filtered:
            continue
        
        df_gen_filtered = pd.DataFrame(df_gen_filtered)
        df_reco_filtered = pd.DataFrame(df_reco_filtered)
        
        label = f"Gen {pdg_name(gen_pid)} → Reco {pdg_name(reco_pid)} (N={count})"
        
        if verbose:
            print(f"    {label}")
        
        # Generar todos los plots para este caso
        plot_energy_ratio(df_gen_filtered, f"{label} - Gen", subdir, pdg_name)
        plot_hit_distributions(df_gen_filtered, f"{label} - Gen", subdir, pdg_name)
        plot_energy_by_subdet(df_gen_filtered, f"{label} - Gen", subdir, pdg_name)
        plot_ecal_hcal_ratio(df_gen_filtered, f"{label} - Gen", subdir, pdg_name)
        plot_hits_ecal_vs_hcal(df_gen_filtered, f"{label} - Gen", subdir, pdg_name)
        
        if not df_reco_filtered.empty:
            plot_energy_ratio(df_reco_filtered, f"{label} - Reco", subdir, pdg_name)
            plot_hit_distributions(df_reco_filtered, f"{label} - Reco", subdir, pdg_name)
            plot_energy_by_subdet(df_reco_filtered, f"{label} - Reco", subdir, pdg_name)
            plot_ecal_hcal_ratio(df_reco_filtered, f"{label} - Reco", subdir, pdg_name)
            plot_hits_ecal_vs_hcal(df_reco_filtered, f"{label} - Reco", subdir, pdg_name)
    
    # 3. Plots agrupados por gen_pid (todos los reco a los que va ese gen, menos exact matches)
    print("\n  [3/6] Generando plots para mismatches agrupados por gen_pid...")
    for gen_pid, reco_list in mismatches_by_gen.items():
        total_count = sum(count for _, count in reco_list)
        subdir = os.path.join(output_dir, "associations", "mismatches_by_gen", 
                              f"gen{gen_pid}")
        os.makedirs(subdir, exist_ok=True)
        
        # Recopilar todas las partículas gen de este tipo que NO matchean exactamente
        df_gen_filtered = []
        df_reco_filtered = []
        
        for _, gen_row in df_gen.iterrows():
            if not gen_row["is_matched"] or gen_row["matched_reco_idx"] == -999:
                continue
            if abs(int(gen_row["pid"])) != gen_pid:
                continue
            
            # Buscar la partícula reco correspondiente
            reco_matches = df_reco[
                (df_reco["event_idx"] == gen_row["event_idx"]) & 
                (df_reco["pfo_id"] == gen_row["matched_reco_idx"])
            ]
            
            if not reco_matches.empty:
                reco_row = reco_matches.iloc[0]
                reco_pid = abs(int(reco_row["pid"]))
                # Excluir exact matches
                if reco_pid != gen_pid:
                    df_gen_filtered.append(gen_row)
                    df_reco_filtered.append(reco_row)
        
        if not df_gen_filtered:
            continue
        
        df_gen_filtered = pd.DataFrame(df_gen_filtered)
        df_reco_filtered = pd.DataFrame(df_reco_filtered)
        
        reco_pids_str = ", ".join([pdg_name(rp) for rp, _ in reco_list])
        label = f"Gen {pdg_name(gen_pid)} → Reco [{reco_pids_str}] (N={total_count})"
        
        if verbose:
            print(f"    {label}")
        
        plot_energy_ratio(df_gen_filtered, f"{label} - Gen", subdir, pdg_name)
        plot_hit_distributions(df_gen_filtered, f"{label} - Gen", subdir, pdg_name)
        plot_energy_by_subdet(df_gen_filtered, f"{label} - Gen", subdir, pdg_name)
        plot_ecal_hcal_ratio(df_gen_filtered, f"{label} - Gen", subdir, pdg_name)
        plot_hits_ecal_vs_hcal(df_gen_filtered, f"{label} - Gen", subdir, pdg_name)
        
        if not df_reco_filtered.empty:
            plot_energy_ratio(df_reco_filtered, f"{label} - Reco", subdir, pdg_name)
            plot_hit_distributions(df_reco_filtered, f"{label} - Reco", subdir, pdg_name)
            plot_energy_by_subdet(df_reco_filtered, f"{label} - Reco", subdir, pdg_name)
            plot_ecal_hcal_ratio(df_reco_filtered, f"{label} - Reco", subdir, pdg_name)
            plot_hits_ecal_vs_hcal(df_reco_filtered, f"{label} - Reco", subdir, pdg_name)
    
    # 4. Plots agrupados por reco_pid (todos los gen que van a ese reco, menos exact matches)
    print("\n  [4/6] Generando plots para mismatches agrupados por reco_pid...")
    for reco_pid, gen_list in mismatches_by_reco.items():
        total_count = sum(count for _, count in gen_list)
        subdir = os.path.join(output_dir, "associations", "mismatches_by_reco", 
                              f"reco{reco_pid}")
        os.makedirs(subdir, exist_ok=True)
        
        # Recopilar todas las partículas gen que matchean con este reco_pid (excepto exact match)
        df_gen_filtered = []
        df_reco_filtered = []
        
        for gen_pid, count in gen_list:
            for _, gen_row in df_gen.iterrows():
                if not gen_row["is_matched"] or gen_row["matched_reco_idx"] == -999:
                    continue
                if abs(int(gen_row["pid"])) != gen_pid:
                    continue
                
                # Buscar la partícula reco correspondiente
                reco_matches = df_reco[
                    (df_reco["event_idx"] == gen_row["event_idx"]) & 
                    (df_reco["pfo_id"] == gen_row["matched_reco_idx"])
                ]
                
                if not reco_matches.empty:
                    reco_row = reco_matches.iloc[0]
                    if abs(int(reco_row["pid"])) == reco_pid:
                        df_gen_filtered.append(gen_row)
                        df_reco_filtered.append(reco_row)
        
        if not df_gen_filtered:
            continue
        
        df_gen_filtered = pd.DataFrame(df_gen_filtered)
        df_reco_filtered = pd.DataFrame(df_reco_filtered)
        
        gen_pids_str = ", ".join([pdg_name(gp) for gp, _ in gen_list])
        label = f"Gen [{gen_pids_str}] → Reco {pdg_name(reco_pid)} (N={total_count})"
        
        if verbose:
            print(f"    {label}")
        
        plot_energy_ratio(df_gen_filtered, f"{label} - Gen", subdir, pdg_name)
        plot_hit_distributions(df_gen_filtered, f"{label} - Gen", subdir, pdg_name)
        plot_energy_by_subdet(df_gen_filtered, f"{label} - Gen", subdir, pdg_name)
        plot_ecal_hcal_ratio(df_gen_filtered, f"{label} - Gen", subdir, pdg_name)
        plot_hits_ecal_vs_hcal(df_gen_filtered, f"{label} - Gen", subdir, pdg_name)
        
        if not df_reco_filtered.empty:
            plot_energy_ratio(df_reco_filtered, f"{label} - Reco", subdir, pdg_name)
            plot_hit_distributions(df_reco_filtered, f"{label} - Reco", subdir, pdg_name)
            plot_energy_by_subdet(df_reco_filtered, f"{label} - Reco", subdir, pdg_name)
            plot_ecal_hcal_ratio(df_reco_filtered, f"{label} - Reco", subdir, pdg_name)
            plot_hits_ecal_vs_hcal(df_reco_filtered, f"{label} - Reco", subdir, pdg_name)
    
    # 5. Plots para gen sin reco (unmatched gen)
    print("\n  [5/6] Generando plots para gen sin reco (unmatched)...")
    for gen_pid, count in unmatched_gen.items():
        subdir = os.path.join(output_dir, "associations", "unmatched_gen", 
                              f"gen{gen_pid}")
        os.makedirs(subdir, exist_ok=True)
        
        # Filtrar partículas gen sin match
        df_gen_filtered = []
        for _, gen_row in df_gen.iterrows():
            if (not gen_row["is_matched"] or gen_row.get("matched_reco_idx", -999) == -999):
                if abs(int(gen_row["pid"])) == gen_pid:
                    df_gen_filtered.append(gen_row)
        
        if not df_gen_filtered:
            continue
        
        df_gen_filtered = pd.DataFrame(df_gen_filtered)
        
        # Usar el tamaño real del DataFrame filtrado para el label
        actual_count = len(df_gen_filtered)
        label = f"Gen {pdg_name(gen_pid)} sin Reco (N={actual_count})"
        
        if verbose:
            print(f"    {label}")
        
        # Generar plots solo para gen (no hay reco)
        plot_energy_ratio(df_gen_filtered, label, subdir, pdg_name)
        plot_hit_distributions(df_gen_filtered, label, subdir, pdg_name)
        plot_energy_by_subdet(df_gen_filtered, label, subdir, pdg_name)
        plot_ecal_hcal_ratio(df_gen_filtered, label, subdir, pdg_name)
        plot_hits_ecal_vs_hcal(df_gen_filtered, label, subdir, pdg_name)
    
    # 6. Plots para reco sin gen (unmatched reco)
    print("\n  [6/6] Generando plots para reco sin gen (unmatched)...")
    for reco_pid, count in unmatched_reco.items():
        subdir = os.path.join(output_dir, "associations", "unmatched_reco", 
                              f"reco{reco_pid}")
        os.makedirs(subdir, exist_ok=True)
        
        # Filtrar partículas reco sin match
        df_reco_filtered = []
        for _, reco_row in df_reco.iterrows():
            if (not reco_row["is_matched"] or reco_row.get("matched_gen_idx", -999) == -999):
                if abs(int(reco_row["pid"])) == reco_pid:
                    df_reco_filtered.append(reco_row)
        
        if not df_reco_filtered:
            continue
        
        df_reco_filtered = pd.DataFrame(df_reco_filtered)
        
        # Usar el tamaño real del DataFrame filtrado para el label
        actual_count = len(df_reco_filtered)
        label = f"Reco {pdg_name(reco_pid)} sin Gen (N={actual_count})"
        
        if verbose:
            print(f"    {label}")
        
        # Generar plots solo para reco (no hay gen)
        plot_energy_ratio(df_reco_filtered, label, subdir, pdg_name)
        plot_hit_distributions(df_reco_filtered, label, subdir, pdg_name)
        plot_energy_by_subdet(df_reco_filtered, label, subdir, pdg_name)
        plot_ecal_hcal_ratio(df_reco_filtered, label, subdir, pdg_name)
        plot_hits_ecal_vs_hcal(df_reco_filtered, label, subdir, pdg_name)
    
    print("\n  ✅ Plots de asociación completados")


def generate_dR_association_plots(df_gen, df_reco, output_dir, verbose=False):
    """
    Genera plots para cada caso de asociación Gen_pid -> Reco_pid usando asociaciones por dR.
    
    Estructura idéntica a generate_association_plots pero usando la carpeta dR_associations.
    """
    if df_gen.empty or df_reco.empty:
        print("\n  No hay datos de gen o reco para generar plots de asociación por dR")
        return
    
    print("\n" + "="*100)
    print("  Generando plots por asociación Gen-Reco (usando dR)")
    print("="*100)
    
    # Identificar casos únicos
    exact_matches, mismatches_individual, mismatches_by_reco, mismatches_by_gen, unmatched_gen, unmatched_reco = get_association_cases(df_gen, df_reco)
    
    print(f"\n  Casos encontrados (dR):")
    print(f"    - Exact matches (aciertos): {len(exact_matches)}")
    print(f"    - Mismatches individuales (parejas únicas): {len(mismatches_individual)}")
    print(f"    - Mismatches agrupados por gen_pid: {len(mismatches_by_gen)}")
    print(f"    - Mismatches agrupados por reco_pid: {len(mismatches_by_reco)}")
    print(f"    - Gen sin reco (unmatched gen): {len(unmatched_gen)} tipos de partícula")
    print(f"    - Reco sin gen (unmatched reco): {len(unmatched_reco)} tipos de partícula")
    
    # 1. Plots para exact matches: gen_pid -> reco_pid (donde gen_pid == reco_pid)
    print("\n  [1/6] Generando plots para exact matches (aciertos) - dR...")
    for (gen_pid, reco_pid), count in exact_matches.items():
        subdir = os.path.join(output_dir, "dR_associations", "exact_matches", 
                              f"gen{gen_pid}_reco{reco_pid}")
        os.makedirs(subdir, exist_ok=True)
        
        # Filtrar gen y reco para este caso específico
        df_gen_filtered = []
        df_reco_filtered = []
        
        for _, gen_row in df_gen.iterrows():
            if not gen_row["is_matched"] or gen_row["matched_reco_idx"] == -999:
                continue
            if abs(int(gen_row["pid"])) != gen_pid:
                continue
            
            # Buscar la partícula reco correspondiente
            reco_matches = df_reco[
                (df_reco["event_idx"] == gen_row["event_idx"]) & 
                (df_reco["pfo_id"] == gen_row["matched_reco_idx"])
            ]
            
            if not reco_matches.empty:
                reco_row = reco_matches.iloc[0]
                if abs(int(reco_row["pid"])) == reco_pid:
                    df_gen_filtered.append(gen_row)
                    df_reco_filtered.append(reco_row)
        
        if not df_gen_filtered:
            continue
        
        df_gen_filtered = pd.DataFrame(df_gen_filtered)
        df_reco_filtered = pd.DataFrame(df_reco_filtered)
        
        label = f"Gen {pdg_name(gen_pid)} → Reco {pdg_name(reco_pid)} (N={count}) [dR]"
        
        if verbose:
            print(f"    {label}")
        
        # Generar todos los plots para este caso
        plot_energy_ratio(df_gen_filtered, f"{label} - Gen", subdir, pdg_name)
        plot_hit_distributions(df_gen_filtered, f"{label} - Gen", subdir, pdg_name)
        plot_energy_by_subdet(df_gen_filtered, f"{label} - Gen", subdir, pdg_name)
        plot_ecal_hcal_ratio(df_gen_filtered, f"{label} - Gen", subdir, pdg_name)
        plot_hits_ecal_vs_hcal(df_gen_filtered, f"{label} - Gen", subdir, pdg_name)
        
        if not df_reco_filtered.empty:
            plot_energy_ratio(df_reco_filtered, f"{label} - Reco", subdir, pdg_name)
            plot_hit_distributions(df_reco_filtered, f"{label} - Reco", subdir, pdg_name)
            plot_energy_by_subdet(df_reco_filtered, f"{label} - Reco", subdir, pdg_name)
            plot_ecal_hcal_ratio(df_reco_filtered, f"{label} - Reco", subdir, pdg_name)
            plot_hits_ecal_vs_hcal(df_reco_filtered, f"{label} - Reco", subdir, pdg_name)
    
    # 2. Plots para mismatches individuales: cada pareja (gen_pid, reco_pid) donde gen_pid != reco_pid
    print("\n  [2/6] Generando plots para mismatches individuales (cada pareja) - dR...")
    for (gen_pid, reco_pid), count in mismatches_individual.items():
        subdir = os.path.join(output_dir, "dR_associations", "mismatches_individual", 
                              f"gen{gen_pid}_reco{reco_pid}")
        os.makedirs(subdir, exist_ok=True)
        
        # Filtrar gen y reco para esta pareja específica
        df_gen_filtered = []
        df_reco_filtered = []
        
        for _, gen_row in df_gen.iterrows():
            if not gen_row["is_matched"] or gen_row["matched_reco_idx"] == -999:
                continue
            if abs(int(gen_row["pid"])) != gen_pid:
                continue
            
            # Buscar la partícula reco correspondiente
            reco_matches = df_reco[
                (df_reco["event_idx"] == gen_row["event_idx"]) & 
                (df_reco["pfo_id"] == gen_row["matched_reco_idx"])
            ]
            
            if not reco_matches.empty:
                reco_row = reco_matches.iloc[0]
                if abs(int(reco_row["pid"])) == reco_pid:
                    df_gen_filtered.append(gen_row)
                    df_reco_filtered.append(reco_row)
        
        if not df_gen_filtered:
            continue
        
        df_gen_filtered = pd.DataFrame(df_gen_filtered)
        df_reco_filtered = pd.DataFrame(df_reco_filtered)
        
        label = f"Gen {pdg_name(gen_pid)} → Reco {pdg_name(reco_pid)} (N={count}) [dR]"
        
        if verbose:
            print(f"    {label}")
        
        # Generar todos los plots para este caso
        plot_energy_ratio(df_gen_filtered, f"{label} - Gen", subdir, pdg_name)
        plot_hit_distributions(df_gen_filtered, f"{label} - Gen", subdir, pdg_name)
        plot_energy_by_subdet(df_gen_filtered, f"{label} - Gen", subdir, pdg_name)
        plot_ecal_hcal_ratio(df_gen_filtered, f"{label} - Gen", subdir, pdg_name)
        plot_hits_ecal_vs_hcal(df_gen_filtered, f"{label} - Gen", subdir, pdg_name)
        
        if not df_reco_filtered.empty:
            plot_energy_ratio(df_reco_filtered, f"{label} - Reco", subdir, pdg_name)
            plot_hit_distributions(df_reco_filtered, f"{label} - Reco", subdir, pdg_name)
            plot_energy_by_subdet(df_reco_filtered, f"{label} - Reco", subdir, pdg_name)
            plot_ecal_hcal_ratio(df_reco_filtered, f"{label} - Reco", subdir, pdg_name)
            plot_hits_ecal_vs_hcal(df_reco_filtered, f"{label} - Reco", subdir, pdg_name)
    
    # 3. Plots agrupados por gen_pid (todos los reco a los que va ese gen, menos exact matches)
    print("\n  [3/6] Generando plots para mismatches agrupados por gen_pid - dR...")
    for gen_pid, reco_list in mismatches_by_gen.items():
        total_count = sum(count for _, count in reco_list)
        subdir = os.path.join(output_dir, "dR_associations", "mismatches_by_gen", 
                              f"gen{gen_pid}")
        os.makedirs(subdir, exist_ok=True)
        
        # Recopilar todas las partículas gen de este tipo que NO matchean exactamente
        df_gen_filtered = []
        df_reco_filtered = []
        
        for _, gen_row in df_gen.iterrows():
            if not gen_row["is_matched"] or gen_row["matched_reco_idx"] == -999:
                continue
            if abs(int(gen_row["pid"])) != gen_pid:
                continue
            
            # Buscar la partícula reco correspondiente
            reco_matches = df_reco[
                (df_reco["event_idx"] == gen_row["event_idx"]) & 
                (df_reco["pfo_id"] == gen_row["matched_reco_idx"])
            ]
            
            if not reco_matches.empty:
                reco_row = reco_matches.iloc[0]
                reco_pid = abs(int(reco_row["pid"]))
                # Excluir exact matches
                if reco_pid != gen_pid:
                    df_gen_filtered.append(gen_row)
                    df_reco_filtered.append(reco_row)
        
        if not df_gen_filtered:
            continue
        
        df_gen_filtered = pd.DataFrame(df_gen_filtered)
        df_reco_filtered = pd.DataFrame(df_reco_filtered)
        
        reco_pids_str = ", ".join([pdg_name(rp) for rp, _ in reco_list])
        label = f"Gen {pdg_name(gen_pid)} → Reco [{reco_pids_str}] (N={total_count}) [dR]"
        
        if verbose:
            print(f"    {label}")
        
        plot_energy_ratio(df_gen_filtered, f"{label} - Gen", subdir, pdg_name)
        plot_hit_distributions(df_gen_filtered, f"{label} - Gen", subdir, pdg_name)
        plot_energy_by_subdet(df_gen_filtered, f"{label} - Gen", subdir, pdg_name)
        plot_ecal_hcal_ratio(df_gen_filtered, f"{label} - Gen", subdir, pdg_name)
        plot_hits_ecal_vs_hcal(df_gen_filtered, f"{label} - Gen", subdir, pdg_name)
        
        if not df_reco_filtered.empty:
            plot_energy_ratio(df_reco_filtered, f"{label} - Reco", subdir, pdg_name)
            plot_hit_distributions(df_reco_filtered, f"{label} - Reco", subdir, pdg_name)
            plot_energy_by_subdet(df_reco_filtered, f"{label} - Reco", subdir, pdg_name)
            plot_ecal_hcal_ratio(df_reco_filtered, f"{label} - Reco", subdir, pdg_name)
            plot_hits_ecal_vs_hcal(df_reco_filtered, f"{label} - Reco", subdir, pdg_name)
    
    # 4. Plots agrupados por reco_pid (todos los gen que van a ese reco, menos exact matches)
    print("\n  [4/6] Generando plots para mismatches agrupados por reco_pid - dR...")
    for reco_pid, gen_list in mismatches_by_reco.items():
        total_count = sum(count for _, count in gen_list)
        subdir = os.path.join(output_dir, "dR_associations", "mismatches_by_reco", 
                              f"reco{reco_pid}")
        os.makedirs(subdir, exist_ok=True)
        
        # Recopilar todas las partículas gen que matchean con este reco_pid (excepto exact match)
        df_gen_filtered = []
        df_reco_filtered = []
        
        for gen_pid, count in gen_list:
            for _, gen_row in df_gen.iterrows():
                if not gen_row["is_matched"] or gen_row["matched_reco_idx"] == -999:
                    continue
                if abs(int(gen_row["pid"])) != gen_pid:
                    continue
                
                # Buscar la partícula reco correspondiente
                reco_matches = df_reco[
                    (df_reco["event_idx"] == gen_row["event_idx"]) & 
                    (df_reco["pfo_id"] == gen_row["matched_reco_idx"])
                ]
                
                if not reco_matches.empty:
                    reco_row = reco_matches.iloc[0]
                    if abs(int(reco_row["pid"])) == reco_pid:
                        df_gen_filtered.append(gen_row)
                        df_reco_filtered.append(reco_row)
        
        if not df_gen_filtered:
            continue
        
        df_gen_filtered = pd.DataFrame(df_gen_filtered)
        df_reco_filtered = pd.DataFrame(df_reco_filtered)
        
        gen_pids_str = ", ".join([pdg_name(gp) for gp, _ in gen_list])
        label = f"Gen [{gen_pids_str}] → Reco {pdg_name(reco_pid)} (N={total_count}) [dR]"
        
        if verbose:
            print(f"    {label}")
        
        plot_energy_ratio(df_gen_filtered, f"{label} - Gen", subdir, pdg_name)
        plot_hit_distributions(df_gen_filtered, f"{label} - Gen", subdir, pdg_name)
        plot_energy_by_subdet(df_gen_filtered, f"{label} - Gen", subdir, pdg_name)
        plot_ecal_hcal_ratio(df_gen_filtered, f"{label} - Gen", subdir, pdg_name)
        plot_hits_ecal_vs_hcal(df_gen_filtered, f"{label} - Gen", subdir, pdg_name)
        
        if not df_reco_filtered.empty:
            plot_energy_ratio(df_reco_filtered, f"{label} - Reco", subdir, pdg_name)
            plot_hit_distributions(df_reco_filtered, f"{label} - Reco", subdir, pdg_name)
            plot_energy_by_subdet(df_reco_filtered, f"{label} - Reco", subdir, pdg_name)
            plot_ecal_hcal_ratio(df_reco_filtered, f"{label} - Reco", subdir, pdg_name)
            plot_hits_ecal_vs_hcal(df_reco_filtered, f"{label} - Reco", subdir, pdg_name)
    
    # 5. Plots para gen sin reco (unmatched gen)
    print("\n  [5/6] Generando plots para gen sin reco (unmatched) - dR...")
    for gen_pid, count in unmatched_gen.items():
        subdir = os.path.join(output_dir, "dR_associations", "unmatched_gen", 
                              f"gen{gen_pid}")
        os.makedirs(subdir, exist_ok=True)
        
        # Filtrar partículas gen sin match
        df_gen_filtered = []
        for _, gen_row in df_gen.iterrows():
            if (not gen_row["is_matched"] or gen_row.get("matched_reco_idx", -999) == -999):
                if abs(int(gen_row["pid"])) == gen_pid:
                    df_gen_filtered.append(gen_row)
        
        if not df_gen_filtered:
            continue
        
        df_gen_filtered = pd.DataFrame(df_gen_filtered)
        
        # Usar el tamaño real del DataFrame filtrado para el label
        actual_count = len(df_gen_filtered)
        label = f"Gen {pdg_name(gen_pid)} sin Reco (N={actual_count}) [dR]"
        
        if verbose:
            print(f"    {label}")
        
        # Generar plots solo para gen (no hay reco)
        plot_energy_ratio(df_gen_filtered, label, subdir, pdg_name)
        plot_hit_distributions(df_gen_filtered, label, subdir, pdg_name)
        plot_energy_by_subdet(df_gen_filtered, label, subdir, pdg_name)
        plot_ecal_hcal_ratio(df_gen_filtered, label, subdir, pdg_name)
        plot_hits_ecal_vs_hcal(df_gen_filtered, label, subdir, pdg_name)
    
    # 6. Plots para reco sin gen (unmatched reco)
    print("\n  [6/6] Generando plots para reco sin gen (unmatched) - dR...")
    for reco_pid, count in unmatched_reco.items():
        subdir = os.path.join(output_dir, "dR_associations", "unmatched_reco", 
                              f"reco{reco_pid}")
        os.makedirs(subdir, exist_ok=True)
        
        # Filtrar partículas reco sin match
        df_reco_filtered = []
        for _, reco_row in df_reco.iterrows():
            if (not reco_row["is_matched"] or reco_row.get("matched_gen_idx", -999) == -999):
                if abs(int(reco_row["pid"])) == reco_pid:
                    df_reco_filtered.append(reco_row)
        
        if not df_reco_filtered:
            continue
        
        df_reco_filtered = pd.DataFrame(df_reco_filtered)
        
        # Usar el tamaño real del DataFrame filtrado para el label
        actual_count = len(df_reco_filtered)
        label = f"Reco {pdg_name(reco_pid)} sin Gen (N={actual_count}) [dR]"
        
        if verbose:
            print(f"    {label}")
        
        # Generar plots solo para reco (no hay gen)
        plot_energy_ratio(df_reco_filtered, label, subdir, pdg_name)
        plot_hit_distributions(df_reco_filtered, label, subdir, pdg_name)
        plot_energy_by_subdet(df_reco_filtered, label, subdir, pdg_name)
        plot_ecal_hcal_ratio(df_reco_filtered, label, subdir, pdg_name)
        plot_hits_ecal_vs_hcal(df_reco_filtered, label, subdir, pdg_name)
    
    print("\n  ✅ Plots de asociación por dR completados")


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Análisis de hits y trazas por tipo de partícula desde archivos ROOT EDM4HEP"
    )
    parser.add_argument("-i", "--input", nargs="+",
                        help="Archivo .root o directorio con archivos .root (no requerido si --from-csv)")
    parser.add_argument("--all", action="store_true",
                        help="Si -i es un directorio, procesar todos los .root")
    parser.add_argument("-o", "--output", default="analysis_output",
                        help="Directorio de salida para plots y CSVs")
    parser.add_argument("--no-plots", action="store_true",
                        help="No generar plots, solo CSV y resumen en consola")
    parser.add_argument("--max-files", type=int, default=None,
                        help="Número máximo de archivos a procesar")
    parser.add_argument("--max-events", type=int, default=None,
                        help="Número máximo de eventos a procesar por archivo")
    parser.add_argument("--from-csv", action="store_true",
                        help="Cargar datos desde CSVs existentes en el directorio de salida")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Muestra información detallada durante el procesamiento")
    args = parser.parse_args()

    # Validar argumentos
    if not args.from_csv and not args.input:
        parser.error("-i/--input es requerido a menos que se use --from-csv")

    os.makedirs(args.output, exist_ok=True)
    gen_csv = os.path.join(args.output, "gen_particles.csv")
    reco_csv = os.path.join(args.output, "reco_particles.csv")
    gen_csv_dR = os.path.join(args.output, "gen_particles_dR.csv")
    reco_csv_dR = os.path.join(args.output, "reco_particles_dR.csv")

    # Modo CSV: cargar desde archivos existentes
    if args.from_csv:
        print("Cargando datos desde CSVs existentes...")
        df_gen_all = pd.read_csv(gen_csv) if os.path.exists(gen_csv) else pd.DataFrame()
        df_reco_all = pd.read_csv(reco_csv) if os.path.exists(reco_csv) else pd.DataFrame()
        df_gen_all_dR = pd.read_csv(gen_csv_dR) if os.path.exists(gen_csv_dR) else pd.DataFrame()
        df_reco_all_dR = pd.read_csv(reco_csv_dR) if os.path.exists(reco_csv_dR) else pd.DataFrame()
        
        if df_gen_all.empty and df_reco_all.empty:
            print(f"Error: No se encontraron CSVs en {args.output}")
            sys.exit(1)
        
        print(f"  Gen: {len(df_gen_all)} partículas")
        print(f"  Reco: {len(df_reco_all)} partículas")
        if not df_gen_all_dR.empty:
            print(f"  Gen (dR): {len(df_gen_all_dR)} partículas")
        if not df_reco_all_dR.empty:
            print(f"  Reco (dR): {len(df_reco_all_dR)} partículas")
    else:
        # Modo normal: procesar archivos ROOT
        # Recopilar archivos
        if os.path.isdir(args.input[0]) and args.all:
            root_files = sorted(glob.glob(os.path.join(args.input[0], "*.root")))
            # También buscar en subdirectorios
            root_files.extend(sorted(glob.glob(os.path.join(args.input[0], "*/*.root"))))
        elif isinstance(args.input, list):
            root_files = [f for f in args.input if os.path.isfile(f) and f.endswith(".root")]
        else:
            print(f"Error: {args.input} no es un archivo .root ni un directorio válido.")
            sys.exit(1)

        if not root_files:
            print("No se encontraron archivos .root")
            sys.exit(1)

        if args.max_files is not None:
            root_files = root_files[:args.max_files]

        print(f"Procesando {len(root_files)} archivo(s)...")

        all_gen = []
        all_reco = []
        all_gen_dR = []
        all_reco_dR = []
        n_gen_pion_list = []
        n_matched_reco_pions = []
        n_matched_reco_pions_w_neutron = []
        pion_efficiencies = []
        pion_efficiencies_neutron = []
        neutral_recover_dfs = []
        for i, fpath in enumerate(root_files, start=1):
            basename = os.path.basename(fpath)
            print(f"\nArchivo {i}/{len(root_files)}: {basename}")
            
            # # Procesar con asociaciones normales (RecoMCTruthLink)
            # try:
            #     df_gen, df_reco = process_root_file(fpath, max_events=args.max_events, verbose=args.verbose)
            #     df_gen["source_file"] = basename
            #     df_reco["source_file"] = basename
            #     all_gen.append(df_gen)
            #     all_reco.append(df_reco)
            # except Exception as e:
            #     print(f"  ⚠ Error procesando {basename}: {e}")
            #     continue
            
            # Procesar con asociaciones por dR
            # try:
            df_gen_dR, df_reco_dR, n_gen_pions, n_matches, n_matches_neutron, pion_efficiency, pion_efficiency_neutron, neutral_recover_df = process_root_file_dR(fpath, max_events=args.max_events, verbose=args.verbose)
            df_gen_dR["source_file"] = basename
            df_reco_dR["source_file"] = basename
            all_gen_dR.append(df_gen_dR)
            all_reco_dR.append(df_reco_dR)
            n_gen_pion_list.append(n_gen_pions)
            n_matched_reco_pions.append(n_matches)
            n_matched_reco_pions_w_neutron.append(n_matches_neutron)
            pion_efficiencies.append(pion_efficiency)
            pion_efficiencies_neutron.append(pion_efficiency_neutron)
            neutral_recover_dfs.append(neutral_recover_df)
            # except Exception as e:
                # print(f"  ⚠ Error procesando {basename} con dR: {e}")
                # continue

        # Concatenar resultados
        df_gen_all = pd.concat(all_gen, ignore_index=True) if all_gen else pd.DataFrame()
        df_reco_all = pd.concat(all_reco, ignore_index=True) if all_reco else pd.DataFrame()
        df_gen_all_dR = pd.concat(all_gen_dR, ignore_index=True) if all_gen_dR else pd.DataFrame()
        df_reco_all_dR = pd.concat(all_reco_dR, ignore_index=True) if all_reco_dR else pd.DataFrame()
        df_neutral_recover = pd.concat(neutral_recover_dfs, ignore_index=True) if neutral_recover_dfs else pd.DataFrame()

        
        # Guardar resúmenes de pion efficiency
        summary_df = pd.DataFrame({
            "source_file": [os.path.basename(f) for f in root_files],
            "n_gen_pions": n_gen_pion_list,
            "n_matched_reco_pions": n_matched_reco_pions,
            "n_matched_reco_pions_w_neutron": n_matched_reco_pions_w_neutron,
            "pion_efficiency": pion_efficiencies,
            "pion_efficiency_neutron": pion_efficiencies_neutron
        })
        summary_csv = os.path.join("neutral_recover_images", "pion_efficiency_summary.csv")
        summary_df.to_csv(summary_csv, index=False)
        print(f"Resumen de pion efficiency guardado: {summary_csv}")
        # Guardar neutral recover details
        if not df_neutral_recover.empty:
            neutral_recover_csv = os.path.join("neutral_recover_images", "neutral_recover_details.csv")
            df_neutral_recover.to_csv(neutral_recover_csv, index=False)
            print(f"Detalles de neutral recover guardados: {neutral_recover_csv}  ({len(df_neutral_recover)} casos)")

        # Guardar CSVs
        if not df_gen_all.empty:
            df_gen_all.to_csv(gen_csv, index=False)
            print(f"\nCSV gen guardado: {gen_csv}  ({len(df_gen_all)} partículas)")
        
        if not df_reco_all.empty:
            df_reco_all.to_csv(reco_csv, index=False)
            print(f"CSV reco guardado: {reco_csv}  ({len(df_reco_all)} partículas)")
        
        if not df_gen_all_dR.empty:
            df_gen_all_dR.to_csv(gen_csv_dR, index=False)
            print(f"CSV gen (dR) guardado: {gen_csv_dR}  ({len(df_gen_all_dR)} partículas)")
        
        if not df_reco_all_dR.empty:
            df_reco_all_dR.to_csv(reco_csv_dR, index=False)
            print(f"CSV reco (dR) guardado: {reco_csv_dR}  ({len(df_reco_all_dR)} partículas)")

    # Resúmenes en consola
    print_summary_table(df_gen_all, "Gen (MCParticles)")
    print_summary_table(df_reco_all, "Reco (PandoraPFOs)")

    # Plots
    if not args.no_plots:
        print("\nGenerando plots...")

        if not df_reco_all.empty:
            plot_energy_ratio(df_reco_all, "Reco Pandora", args.output, pdg_name)
            plot_hit_distributions(df_reco_all, "Reco Pandora", args.output, pdg_name)
            plot_energy_by_subdet(df_reco_all, "Reco Pandora", args.output, pdg_name)
            plot_ecal_hcal_ratio(df_reco_all, "Reco Pandora", args.output, pdg_name)
            plot_hits_ecal_vs_hcal(df_reco_all, "Reco Pandora", args.output, pdg_name)
        if not df_gen_all.empty:
            plot_energy_ratio(df_gen_all, "Gen", args.output, pdg_name)
            plot_hit_distributions(df_gen_all, "Gen", args.output, pdg_name)
            plot_energy_by_subdet(df_gen_all, "Gen", args.output, pdg_name)
            plot_ecal_hcal_ratio(df_gen_all, "Gen", args.output, pdg_name)
            plot_hits_ecal_vs_hcal(df_gen_all, "Gen", args.output, pdg_name)
        
        # Generar plots por asociación Gen-Reco
        generate_association_plots(df_gen_all, df_reco_all, args.output, verbose=args.verbose)
        
        # Generar plots por asociación Gen-Reco usando dR
        if not df_gen_all_dR.empty and not df_reco_all_dR.empty:
            generate_dR_association_plots(df_gen_all_dR, df_reco_all_dR, args.output, verbose=args.verbose)

    print("\n✅ Análisis completado.")


if __name__ == "__main__":
    main()
