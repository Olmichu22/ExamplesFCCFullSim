"""
Versión paralela de analysisRHOTree.py.
Divide los ficheros de entrada en chunks y los procesa en paralelo usando
ProcessPoolExecutor con contexto fork.  Cada worker escribe un ROOT parcial;
el proceso principal los fusiona con hadd.

USO:
    python analysisRHOTree_parallel.py [mismos args que analysisRHOTree.py] --n-workers N
"""

import ctypes
import logging
import math
import multiprocessing
import os
import pickle
import pprint
import subprocess
import sys
import time
from array import array
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
import ROOT
import yaml
from podio import root_io
from ROOT import TFile, TTree, TH1F, TH2F
import edm4hep

from modules.TauDecays import extractTauDecays
from modules import (ParticleObjects, electronReco, muonReco, myutils, pi0Reco,
                     tauReco, particleMatch)
from modules import weightsPol
from modules import optimalVariabRho


# ── Config ────────────────────────────────────────────────────────────────────

_DEFAULT_CONFIG = "config/default/taurecolong.yaml"
_OUTPUT_BASE    = "Results/RhoAnalysis/"

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
        filename=log_file,
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        force=True,
    )
    lg = logging.getLogger(f"worker_{worker_id}")
    return {
        "processing": lg,
        "io":         lg,
        "config":     lg,
        "pi0mass":    lg,
    }


def _create_trees_and_branches(outfile, root_histograms_super):
    """Crea TTrees y branches para cada key en root_histograms_super."""
    outfile.cd()
    trees = {}
    branches_super = {}
    tree_name = "outtree"
    for key in root_histograms_super:
        trees[key] = ROOT.TTree(tree_name + f"_{key}", f"processed variables - {key}")
        branches_super[key] = {}
        for var in _VARIABS:
            branches_super[key][var] = ctypes.c_double(0.0)
            trees[key].Branch(var, ctypes.addressof(branches_super[key][var]), f"{var}/D")
        for var in _VECTOR_VARIABS:
            branches_super[key][var] = ROOT.std.vector("double")()
            trees[key].Branch(var, branches_super[key][var])
    return trees, branches_super


# ── Worker ────────────────────────────────────────────────────────────────────

def process_chunk_stage1(filenames_chunk, mlpf_chunk, global_event_offset,
                         config_bundle, worker_id):
    """
    Procesa un chunk de ficheros ROOT y escribe un fichero ROOT parcial.
    Devuelve (partial_path, totalEvents, selectedEvents, sumWeights, sumWeightsP1, sumWeightsM1).
    """
    outputpath    = config_bundle["outputpath"]
    loggers       = _setup_worker_logging(outputpath, worker_id)
    logger_process = loggers["processing"]
    logger_io      = loggers["io"]

    # Extraer parámetros del bundle
    dRMax          = config_bundle["dRMax"]
    tauPCut        = config_bundle["tauPCut"]
    minPTauPhoton  = config_bundle["minPTauPhoton"]
    minPTauPion    = config_bundle["minPTauPion"]
    PNeutron       = config_bundle["PNeutron"]
    generalPCut    = config_bundle["generalPCut"]
    photon_config  = config_bundle["photon_config"]
    test_extremes  = config_bundle["test_extremes"]
    selectDecay    = config_bundle["selectDecay"]
    sample         = config_bundle["sample"]
    gatr_results_path = config_bundle["gatr_results_path"]
    sin_eff        = config_bundle["sin_eff"]
    test_pfo       = config_bundle["test_pfo"]
    fileOutName_base = config_bundle["fileOutName_base"]
    histogram_config = config_bundle["histogram_config"]
    gen_taus_sample  = config_bundle.get("gen_taus_sample", False)
    
    minPTauElectron = config_bundle.get("minPTauElectron", 10.0)
    minPTauMuon     = config_bundle.get("minPTauMuon", 10.0)

    genparts  = "MCParticles"
    pfobjects = "PandoraPFOs"

    logger_io.info("Worker %d: procesando %d ficheros", worker_id, len(filenames_chunk))

    # Crear histogramas ROOT dentro del worker (después del fork — seguro)
    root_histograms = myutils.set_up_root_histograms(histogram_config)
    root_histograms_super = {"original": root_histograms}
    if test_extremes:
        if photon_config.get("energy", {}):
            root_histograms_super["energy_max_err"] = myutils.clone_histograms_with_suffix(root_histograms, "_energy_max")
            root_histograms_super["energy_min_err"] = myutils.clone_histograms_with_suffix(root_histograms, "_energy_min")
        if photon_config.get("direction", {}):
            root_histograms_super["direction_max_err"] = myutils.clone_histograms_with_suffix(root_histograms, "_direction_max")
            # root_histograms_super["direction_min_err"] = myutils.clone_histograms_with_suffix(root_histograms, "_direction_min")

    # Fichero ROOT de salida parcial
    partial_path = os.path.join(outputpath, f"partial_worker{worker_id}_{fileOutName_base}.root")
    partial_outfile = ROOT.TFile(partial_path, "RECREATE")

    trees, branches_super = _create_trees_and_branches(partial_outfile, root_histograms_super)

    weight       = 1.0
    totalEvents  = 0
    selectedEvents = 0
    sumWeights   = 0.0
    sumWeightsP1 = 0.0
    sumWeightsM1 = 0.0

    # ── Event loop ────────────────────────────────────────────────────────────
    eventid = 0
    skipped_files = []
    for filename in filenames_chunk:
        try:
            file_reader = root_io.Reader([filename])
        except Exception as _file_exc:
            logger_io.warning("Worker %d: skipping unreadable file %s: %s",
                              worker_id, filename, _file_exc)
            skipped_files.append(filename)
            continue
        for event in file_reader.get("events"):
            logger_process.debug("Processing event %d", totalEvents)
            if totalEvents % 500 == 0:
                logger_process.info("Worker %d: processed %d events", worker_id, totalEvents)

            for tree_key in trees:
                for var in _VECTOR_VARIABS:
                    branches_super[tree_key][var].clear()

            prevSelectedEvents = selectedEvents
            totalEvents += 1

            mc_particles = event.get(genparts)
            beamE = mc_particles[0].getEnergy()

            if gen_taus_sample:
                genTaus  = tauReco.findAllGenTaus(mc_particles)
                nGenTaus = len(genTaus)
                gen_taus = True
            else:
                gen_taus = False
                genTaus  = {}
                nGenTaus = 0

            GenZMass    = 0
            GenZVisMass = 0
            if nGenTaus == 2:
                ZGenP       = genTaus[0].getMomentum() + genTaus[1].getMomentum()
                ZVisPGen    = genTaus[0].getvisMomentum() + genTaus[1].getvisMomentum()
                GenZMass    = ZGenP.M()
                GenZVisMass = ZVisPGen.M()

            pfos = event.get(pfobjects)
            recoTau, recoElectrons, recoMuons, tau_extremes = extractTauDecays(
                gatr_results_path,
                mlpf_chunk,
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

            recoTaus_extremes = {"original": recoTau}
            if tau_extremes["energy"]["max"] is not None:
                recoTaus_extremes["energy_max_err"] = tau_extremes["energy"]["max"]
                recoTaus_extremes["energy_min_err"] = tau_extremes["energy"]["min"]
            if tau_extremes["direction"]["max"] is not None:
                recoTaus_extremes["direction_max_err"] = tau_extremes["direction"]["max"]
                # recoTaus_extremes["direction_min_err"] = tau_extremes["direction"]["min"]

            for tree_key in trees:
                recoTau       = recoTaus_extremes[tree_key]
                branches      = branches_super[tree_key]
                new_tree      = trees[tree_key]
                root_histograms = root_histograms_super[tree_key]

                unsorted_recoTaus = recoTau
                recoTaus          = myutils.sort_by_P(unsorted_recoTaus)
                nRecoTaus         = len(recoTaus)

                countMuonsP10    = 0
                countElectronsP10 = 0
                if nRecoTaus < 1:
                    continue

                for mu in recoMuons:
                    muonP4 = recoMuons[mu].getMomentum()
                    if muonP4.P() > minPTauMuon:
                        countMuonsP10 += 1

                for e in recoElectrons:
                    electronP4 = recoElectrons[e].getMomentum()
                    if electronP4.P() > minPTauElectron:
                        countElectronsP10 += 1

                method = "MaxP"
                if method == "MaxP":
                    idx = 0
                else:
                    for i, tau in enumerate(recoTaus):
                        if tau.getID() == selectDecay:
                            idx = i
                            break

                recoTau   = recoTaus[idx]
                recoTauID = recoTau.getID()

                if recoTauID != selectDecay:
                    continue

                recoMesonP4   = recoTau.getMomentum()
                recoTauID     = recoTau.getID()
                recoTauQ      = recoTau.getCharge()
                recoTauConsts = recoTau.getDaughters()
                recoPionP4 = ROOT.TLorentzVector(); recoPionP4.SetXYZM(0, 0, 0, 0)
                for const in recoTauConsts:
                    const_PDG = abs(recoTauConsts[const].getPDG())
                    if const_PDG == 211:
                        recoPion   = recoTauConsts[const]
                        recoPionP4 = ROOT.TLorentzVector()
                        try:
                            recoPionP4.SetXYZM(recoPion.getMomentum().x,
                                               recoPion.getMomentum().y,
                                               recoPion.getMomentum().z,
                                               recoPion.getMass())
                        except Exception:
                            recoPionP4.SetXYZM(recoPion.getMomentum().X(),
                                               recoPion.getMomentum().Y(),
                                               recoPion.getMomentum().Z(),
                                               recoPion.getMass())
                        break

                if recoMesonP4.P() < tauPCut:
                    continue

                if not (nRecoTaus == 1 and
                        ((countElectronsP10 == 1) ^ (countMuonsP10 == 1))):
                    continue

                is_electron = True if countElectronsP10 == 1 else False
                lepP4  = None
                lepPDG = 0
                if is_electron:
                    for e in recoElectrons:
                        electronP4 = recoElectrons[e].getMomentum()
                        if electronP4.P() > minPTauElectron:
                            lepP4  = electronP4
                            lepPDG = 11
                            break
                else:
                    for mu in recoMuons:
                        muonP4 = recoMuons[mu].getMomentum()
                        if muonP4.P() > minPTauMuon:
                            lepP4  = muonP4
                            lepPDG = 13
                            break

                ZMass = (recoMesonP4 + lepP4).M()

                genIndex = -1
                closestDR = 10
                for g in genTaus.keys():
                    genP4 = genTaus[g].getMomentum()
                    dR    = myutils.dRAngle(genP4, recoMesonP4)
                    if dR < closestDR:
                        closestDR = dR
                        genIndex  = g

                if genIndex == -1 and gen_taus:
                    continue

                lepton_genIndex  = -1
                closestDRLepton  = 5
                for g in genTaus.keys():
                    genP4 = genTaus[g].getMomentum()
                    dR    = myutils.dRAngle(genP4, lepP4)
                    if dR < closestDRLepton:
                        closestDRLepton = dR
                        lepton_genIndex = g

                if gen_taus:
                    genMesonP4 = genTaus[genIndex].getvisMomentum()
                    genRhoP4   = genMesonP4
                    genTauID   = genTaus[genIndex].getID()
                    genTauP4   = genTaus[genIndex].getMomentum()
                    genTauConst = genTaus[genIndex].getDaughters()

                    genPionP4 = ROOT.TLorentzVector(); genPionP4.SetXYZM(0, 0, 0, 0)
                    for const in genTauConst:
                        const_PDG = abs(genTauConst[const].getPDG())
                        if const_PDG == 211:
                            genPion   = genTauConst[const]
                            genPionP4 = ROOT.TLorentzVector()
                            try:
                                genPionP4.SetXYZM(genPion.getMomentum().x,
                                                  genPion.getMomentum().y,
                                                  genPion.getMomentum().z,
                                                  genPion.getMass())
                            except Exception:
                                genPionP4.SetXYZM(genPion.getMomentum().X(),
                                                  genPion.getMomentum().Y(),
                                                  genPion.getMomentum().Z(),
                                                  genPion.getMass())
                            break

                    genLepP4     = ROOT.TLorentzVector()
                    genLepP4.SetXYZM(0, 0, 0, 0)
                    genLepTauP4  = ROOT.TLorentzVector()
                    genLepTauP4.SetXYZM(0, 0, 0, 0)
                    genLepPDG = 0
                    if nGenTaus >= 2:
                        lepton_index = genTaus[lepton_genIndex]
                        genLepP4    = lepton_index.getvisMomentum()
                        genLepTauP4 = lepton_index.getMomentum()
                        genLepPDG   = abs(genTaus[lepton_genIndex].getID())
                else:
                    genMesonP4 = ROOT.TLorentzVector(); genMesonP4.SetXYZM(0, 0, 0, 0)
                    genTauP4   = ROOT.TLorentzVector(); genTauP4.SetXYZM(0, 0, 0, 0)
                    genPionP4  = ROOT.TLorentzVector(); genPionP4.SetXYZM(0, 0, 0, 0)
                    genRhoP4   = ROOT.TLorentzVector(); genRhoP4.SetXYZM(0, 0, 0, 0)
                    genLepP4    = ROOT.TLorentzVector(); genLepP4.SetXYZM(0, 0, 0, 0)
                    genLepTauP4 = ROOT.TLorentzVector(); genLepTauP4.SetXYZM(0, 0, 0, 0)
                    genTauID    = -1
                    genTauConst = {}
                    genLepPDG   = 0

                if gen_taus:
                    (gen_cos_theta, gen_cos_psi, gen_cos_beta, gen_w,
                     weight_P1, weight_M1) = optimalVariabRho.wVariab(
                        genTauP4, genMesonP4, genPionP4, beamE, sin_eff=sin_eff)
                    gen_cos_theta_tau = math.cos(genTauP4.Theta())
                else:
                    gen_cos_theta = gen_cos_psi = gen_cos_beta = gen_w = 0
                    weight_P1 = weight_M1 = gen_cos_theta_tau = 0

                if abs(int(genLepPDG)) in [11, 13] and genLepTauP4.E() > 0:
                    weight_lep_P1 = weightsPol.newAtauLep(genLepP4, genLepTauP4, beamE, +1, sin_eff=sin_eff)
                    weight_lep_M1 = weightsPol.newAtauLep(genLepP4, genLepTauP4, beamE, -1, sin_eff=sin_eff)
                else:
                    weight_lep_P1 = 1.0
                    weight_lep_M1 = 1.0

                (cos_theta, cos_psi, cos_beta, w) = optimalVariabRho.wVariabRECO(
                    recoMesonP4, recoPionP4, beamE)
                cos_theta_rho = math.cos(recoMesonP4.Theta())

                # Photons
                if gen_taus:
                    nPhotonsGen = 0.0
                    for const in genTauConst:
                        pdg = abs(genTauConst[const].getPDG())
                        if pdg == 111:
                            pi0const = genTauConst[const].getDaughters()
                            for photon_const in pi0const:
                                if photon_const.getGeneratorStatus() != 1:
                                    continue
                                nPhotonsGen += 1
                                photonP4 = ROOT.TLorentzVector()
                                try:
                                    photonP4.SetXYZM(photon_const.getMomentum().x,
                                                     photon_const.getMomentum().y,
                                                     photon_const.getMomentum().z,
                                                     photon_const.getMass())
                                except Exception:
                                    photonP4.SetXYZM(photon_const.getMomentum().X(),
                                                     photon_const.getMomentum().Y(),
                                                     photon_const.getMomentum().Z(),
                                                     photon_const.getMass())
                                branches["gen_photons_E"].push_back(photonP4.E())
                                branches["gen_photons_theta"].push_back(photonP4.Theta())
                                branches["gen_photons_phi"].push_back(photonP4.Phi())
                else:
                    nPhotonsGen = 0.0
                    branches["gen_photons_E"].push_back(0.0)
                    branches["gen_photons_theta"].push_back(0.0)
                    branches["gen_photons_phi"].push_back(0.0)

                nPhotonsReco = 0.0
                for const in recoTauConsts:
                    pdg = abs(recoTauConsts[const].getPDG())
                    if pdg == 22:
                        nPhotonsReco += 1
                        photon   = recoTauConsts[const]
                        photonP4 = ROOT.TLorentzVector()
                        try:
                            photonP4.SetXYZM(photon.getMomentum().x,
                                             photon.getMomentum().y,
                                             photon.getMomentum().z,
                                             photon.getMass())
                        except Exception:
                            photonP4.SetXYZM(photon.getMomentum().X(),
                                             photon.getMomentum().Y(),
                                             photon.getMomentum().Z(),
                                             photon.getMass())
                        branches["reco_photons_E"].push_back(photonP4.E())
                        branches["reco_photons_theta"].push_back(photonP4.Theta())
                        branches["reco_photons_phi"].push_back(photonP4.Phi())

                if abs(cos_theta) == 1:
                    continue
                if abs(cos_psi) == 1:
                    continue

                if selectedEvents == prevSelectedEvents:
                    selectedEvents += 1

                # Fill branches
                branches["genTauP"].value         = genTauP4.P()
                branches["genTauTheta"].value      = genTauP4.Theta()
                branches["genTauPhi"].value        = genTauP4.Phi()
                branches["genTauE"].value          = genTauP4.E()
                branches["genTauM"].value          = genTauP4.M()
                branches["genMesonP"].value        = genMesonP4.P()
                branches["genMesonTheta"].value    = genMesonP4.Theta()
                branches["genMesonPhi"].value      = genMesonP4.Phi()
                branches["genMesonE"].value        = genMesonP4.E()
                branches["genMesonM"].value        = genMesonP4.M()
                branches["genPionP"].value         = genPionP4.P()
                branches["genPionTheta"].value     = genPionP4.Theta()
                branches["genPionPhi"].value       = genPionP4.Phi()
                branches["genPionE"].value         = genPionP4.E()
                branches["genPionM"].value         = genPionP4.M()
                branches["gen_cos_theta"].value    = gen_cos_theta
                branches["gen_cos_psi"].value      = gen_cos_psi
                branches["gen_cos_beta"].value     = gen_cos_beta
                branches["gen_w"].value            = gen_w
                branches["genOmega"].value         = gen_w
                branches["weight_P1"].value        = weight_P1
                branches["weight_M1"].value        = weight_M1
                branches["genTauID"].value         = genTauID
                branches["gen_cos_theta_tau"].value = gen_cos_theta_tau
                branches["recoMesonP"].value       = recoMesonP4.P()
                branches["recoMesonTheta"].value   = recoMesonP4.Theta()
                branches["recoMesonPhi"].value     = recoMesonP4.Phi()
                branches["recoPionP"].value        = recoPionP4.P()
                branches["recoPionTheta"].value    = recoPionP4.Theta()
                branches["recoPionPhi"].value      = recoPionP4.Phi()
                branches["recoMesonE"].value       = recoMesonP4.E()
                branches["recoMesonM"].value       = recoMesonP4.M()
                branches["recoPionE"].value        = recoPionP4.E()
                branches["recoPionM"].value        = recoPionP4.M()
                branches["cos_theta"].value        = cos_theta
                branches["cos_psi"].value          = cos_psi
                branches["cos_beta"].value         = cos_beta
                branches["omega"].value            = w
                branches["recoTauID"].value        = recoTauID
                branches["cos_theta_rho"].value    = cos_theta_rho
                branches["ZMass"].value            = ZMass
                branches["GenZMass"].value         = GenZMass
                branches["GenZVisMass"].value      = GenZVisMass
                branches["beamE"].value            = beamE
                branches["nPhotonsReco"].value     = nPhotonsReco
                branches["nPhotonsGen"].value      = nPhotonsGen
                branches["isElectron"].value       = float(is_electron)
                branches["lepP"].value             = lepP4.P()
                branches["lepE"].value             = lepP4.E()
                branches["lepTheta"].value         = lepP4.Theta()
                branches["lepPhi"].value           = lepP4.Phi()
                branches["lepPDG"].value           = lepPDG
                branches["genLepP"].value          = genLepP4.P()
                branches["genLepE"].value          = genLepP4.E()
                branches["genLepTheta"].value      = genLepP4.Theta()
                branches["genLepPhi"].value        = genLepP4.Phi()
                branches["genLepPDG"].value        = genLepPDG
                branches["genLepTauP"].value       = genLepTauP4.P()
                branches["genLepTauE"].value       = genLepTauP4.E()
                branches["genLepTauTheta"].value   = genLepTauP4.Theta()
                branches["genLepTauPhi"].value     = genLepTauP4.Phi()
                branches["genLepTauM"].value       = genLepTauP4.M()
                branches["weight_lep_P1"].value    = weight_lep_P1
                branches["weight_lep_M1"].value    = weight_lep_M1
                branches["genOptimalvarPi"].value     = genPionP4.E()/genTauP4.E() if genTauP4.E() > 0 else 0.0
                branches["genOptimalvarLep"].value    = genLepP4.E()/genLepTauP4.E() if genLepTauP4.E() > 0 else 0.0
                branches["recoOptimalvarPi"].value     = genPionP4.E()/genTauP4.E() if genTauP4.E() > 0 else 0.0
                branches["recoOptimalvarLep"].value    = genLepP4.E()/genLepTauP4.E() if genLepTauP4.E() > 0 else 0.0
                new_tree.Fill()

                # Histogramas ALL
                x = 2 * recoMesonP4.E() / beamE - 1
                root_histograms["Reco"]["Events"]["RecoMesonEOverBeamE_ALL"].Fill(recoMesonP4.E() / beamE, weight)
                root_histograms["Reco"]["Events"]["RecoMesonCosTheta_ALL"].Fill(math.cos(recoMesonP4.Theta()), weight)
                root_histograms["Reco"]["Events"]["CosTheta_ALL"].Fill(cos_theta, weight)
                root_histograms["Reco"]["Events"]["CosPsi_ALL"].Fill(cos_psi, weight)
                root_histograms["Gen"]["Events"]["Omega_GEN_ALL"].Fill(gen_w, weight)
                root_histograms["Gen"]["Events"]["CosTheta_GEN_ALL"].Fill(gen_cos_theta, weight)
                root_histograms["Gen"]["Events"]["CosPsi_GEN_ALL"].Fill(gen_cos_psi, weight)
                root_histograms["Gen"]["Events"]["CosThetaTau_GEN_ALL"].Fill(math.cos(genTauP4.Theta()), weight)
                root_histograms["Gen"]["Events"]["CosThetaRho_GEN_ALL"].Fill(math.cos(genRhoP4.Theta()), weight)
                root_histograms["Gen"]["Events"]["CosThetaRho_ALL"].Fill(math.cos(recoMesonP4.Theta()), weight)
                root_histograms["Reco"]["Events"]["OmegaCosTheta_ALL"].Fill(w, cos_theta_rho, weight)
                root_histograms["Gen"]["Events"]["OmegaCosThetaTau_GEN_ALL"].Fill(gen_w, gen_cos_theta_tau, weight)
                root_histograms["Reco"]["Events"]["RecoMeson_X_ALL"].Fill(x, weight)

                if genIndex != -1:
                    selectGEN = selectDecay
                    if selectDecay == 2:
                        selectGEN = 1

                    if genTauID == selectGEN:
                        weight_P1 = weightsPol.newAtauRHO(genTauP4, genMesonP4, beamE,
                                                          genTauConst, genTauID, +1)
                        weight_M1 = weightsPol.newAtauRHO(genTauP4, genMesonP4, beamE,
                                                          genTauConst, genTauID, -1)
                        root_histograms["Matched"]["Events"]["MesonEOverBeamE"].Fill(genMesonP4.E() / beamE, weight)
                        root_histograms["Matched"]["Events"]["MesonEOverBeamE_P1"].Fill(genMesonP4.E() / beamE, weight_P1 * weight)
                        root_histograms["Matched"]["Events"]["MesonEOverBeamE_M1"].Fill(genMesonP4.E() / beamE, weight_M1 * weight)
                        root_histograms["Reco"]["Events"]["Omega_SIGNAL"].Fill(w, weight)
                        root_histograms["Reco"]["Events"]["Omega_SIGNAL_P1"].Fill(w, weight * weight_P1)
                        root_histograms["Reco"]["Events"]["Omega_SIGNAL_M1"].Fill(w, weight * weight_M1)
                        root_histograms["Gen"]["Events"]["Omega_GEN_SIGNAL"].Fill(gen_w, weight)
                        root_histograms["Gen"]["Events"]["Omega_GEN_SIGNAL_P1"].Fill(gen_w, weight * weight_P1)
                        root_histograms["Gen"]["Events"]["Omega_GEN_SIGNAL_M1"].Fill(gen_w, weight * weight_M1)
                        root_histograms["Reco"]["Events"]["OmegaCosTheta_SIGNAL"].Fill(w, cos_theta_rho, weight)
                        root_histograms["Reco"]["Events"]["OmegaCosTheta_SIGNAL_P1"].Fill(w, cos_theta_rho, weight * weight_P1)
                        root_histograms["Reco"]["Events"]["OmegaCosTheta_SIGNAL_M1"].Fill(w, cos_theta_rho, weight * weight_M1)
                        root_histograms["Gen"]["Events"]["OmegaCosThetaTau_GEN_SIGNAL"].Fill(gen_w, gen_cos_theta_tau, weight)
                        root_histograms["Gen"]["Events"]["OmegaCosThetaTau_GEN_SIGNAL_P1"].Fill(gen_w, gen_cos_theta_tau, weight * weight_P1)
                        root_histograms["Gen"]["Events"]["OmegaCosThetaTau_GEN_SIGNAL_M1"].Fill(gen_w, gen_cos_theta_tau, weight * weight_M1)
                        root_histograms["Reco"]["Events"]["CosTheta_SIGNAL"].Fill(cos_theta, weight)
                        root_histograms["Reco"]["Events"]["CosTheta_SIGNAL_P1"].Fill(cos_theta, weight_P1 * weight)
                        root_histograms["Reco"]["Events"]["CosTheta_SIGNAL_M1"].Fill(cos_theta, weight_M1 * weight)
                        root_histograms["Gen"]["Events"]["CosTheta_GEN_SIGNAL"].Fill(cos_theta, weight)
                        root_histograms["Gen"]["Events"]["CosTheta_GEN_SIGNAL_P1"].Fill(cos_theta, weight_P1 * weight)
                        root_histograms["Gen"]["Events"]["CosTheta_GEN_SIGNAL_M1"].Fill(cos_theta, weight_M1 * weight)
                        root_histograms["Reco"]["Events"]["CosPsi_SIGNAL"].Fill(cos_psi, weight)
                        root_histograms["Reco"]["Events"]["CosPsi_SIGNAL_P1"].Fill(cos_psi, weight_P1 * weight)
                        root_histograms["Reco"]["Events"]["CosPsi_SIGNAL_M1"].Fill(cos_psi, weight_M1 * weight)
                        root_histograms["Gen"]["Events"]["CosPsi_GEN_SIGNAL"].Fill(gen_cos_psi, weight)
                        root_histograms["Gen"]["Events"]["CosPsi_GEN_SIGNAL_P1"].Fill(gen_cos_psi, weight_P1 * weight)
                        root_histograms["Gen"]["Events"]["CosPsi_GEN_SIGNAL_M1"].Fill(gen_cos_psi, weight_M1 * weight)
                        root_histograms["Matched"]["Events"]["MesonCosTheta"].Fill(math.cos(genMesonP4.Theta()), weight)
                        root_histograms["Matched"]["Events"]["MesonCosTheta_P1"].Fill(math.cos(genMesonP4.Theta()), weight_P1 * weight)
                        root_histograms["Matched"]["Events"]["MesonCosTheta_M1"].Fill(math.cos(genMesonP4.Theta()), weight_M1 * weight)
                        root_histograms["Reco"]["Events"]["RecoMesonEOverBeamE_SIGNAL"].Fill(recoMesonP4.E() / beamE, weight)
                        root_histograms["Reco"]["Events"]["RecoMesonEOverBeamE_SIGNAL_P1"].Fill(recoMesonP4.E() / beamE, weight_P1 * weight)
                        root_histograms["Reco"]["Events"]["RecoMesonEOverBeamE_SIGNAL_M1"].Fill(recoMesonP4.E() / beamE, weight_M1 * weight)
                        root_histograms["Reco"]["Events"]["RecoMeson_X"].Fill(x, weight)
                        root_histograms["Reco"]["Events"]["RecoMeson_X_P1"].Fill(x, weight * weight_P1)
                        root_histograms["Reco"]["Events"]["RecoMeson_X_M1"].Fill(x, weight * weight_M1)
                        root_histograms["Reco"]["Events"]["RecoMesonCosTheta_SIGNAL"].Fill(math.cos(recoMesonP4.Theta()), weight)
                        root_histograms["Reco"]["Events"]["RecoMesonCosTheta_SIGNAL_P1"].Fill(math.cos(recoMesonP4.Theta()), weight_P1 * weight)
                        root_histograms["Reco"]["Events"]["RecoMesonCosTheta_SIGNAL_M1"].Fill(math.cos(recoMesonP4.Theta()), weight_M1 * weight)
                        root_histograms["Gen"]["Events"]["MesonType"].Fill(genTauID, weight)
                        root_histograms["Reco"]["Events"]["RecoMesonType"].Fill(recoTauID, weight)

                        if selectDecay == 2:
                            cum_P4 = ROOT.TLorentzVector()
                            cum_P4.SetXYZM(0, 0, 0, 0)
                            for const in recoTauConsts:
                                pdg = abs(recoTauConsts[const].getPDG())
                                if pdg == 22:
                                    photon   = recoTauConsts[const]
                                    photonP4 = ROOT.TLorentzVector()
                                    try:
                                        photonP4.SetXYZM(photon.getMomentum().x,
                                                         photon.getMomentum().y,
                                                         photon.getMomentum().z,
                                                         photon.getMass())
                                    except Exception:
                                        photonP4.SetXYZM(photon.getMomentum().X(),
                                                         photon.getMomentum().Y(),
                                                         photon.getMomentum().Z(),
                                                         photon.getMass())
                                    cum_P4 += photonP4
                            recoPi0Mass = cum_P4.M()
                            root_histograms["Reco"]["Events"]["Pi0Mass_SIGNAL"].Fill(recoPi0Mass)

                            cum_P4_gen     = ROOT.TLorentzVector(); cum_P4_gen.SetXYZM(0, 0, 0, 0)
                            cum_P4_no_smear = ROOT.TLorentzVector(); cum_P4_no_smear.SetXYZM(0, 0, 0, 0)
                            pi0 = None
                            for const in genTauConst:
                                pdg = abs(genTauConst[const].getPDG())
                                if pdg == 111:
                                    pi0 = genTauConst[const]
                                    break
                            if pi0 is not None:
                                for photon in pi0.getDaughters():
                                    photonP4 = ROOT.TLorentzVector()
                                    try:
                                        photonP4.SetXYZM(photon.getMomentum().x,
                                                         photon.getMomentum().y,
                                                         photon.getMomentum().z,
                                                         photon.getMass())
                                    except Exception:
                                        photonP4.SetXYZM(photon.getMomentum().X(),
                                                         photon.getMomentum().Y(),
                                                         photon.getMomentum().Z(),
                                                         photon.getMass())
                                    photon_E = photonP4.E()
                                    theta    = photonP4.Theta()
                                    phi      = photonP4.Phi()
                                    sigma    = photon_E * 0.16 / math.sqrt(photon_E)
                                    newE     = float(tauReco.normal_sample(mean=photon_E, stddev=sigma)[0])
                                    px = newE * np.sin(theta) * np.cos(phi)
                                    py = newE * np.sin(theta) * np.sin(phi)
                                    pz = newE * np.cos(theta)
                                    smeared = ROOT.TLorentzVector()
                                    smeared.SetPxPyPzE(px, py, pz, newE)
                                    cum_P4_gen     += smeared
                                    cum_P4_no_smear += photonP4
                                genPi0Mass = cum_P4_gen.M()
                                root_histograms["Gen"]["Events"]["Pi0Mass_GEN_SIGNAL"].Fill(genPi0Mass)

                    else:  # BG
                        root_histograms["Reco"]["Events"]["RecoMesonEOverBeamE_BG"].Fill(recoMesonP4.E() / beamE, weight)
                        root_histograms["Reco"]["Events"]["RecoMesonCosTheta_BG"].Fill(math.cos(recoMesonP4.Theta()), weight)
                        root_histograms["Matched"]["Events"]["MesonEOverBeamE_BG"].Fill(genMesonP4.E() / beamE, weight)
                        root_histograms["Matched"]["Events"]["MesonCosTheta_BG"].Fill(math.cos(genMesonP4.Theta()), weight)
                        root_histograms["Reco"]["Events"]["RecoMeson_X_BG"].Fill(x, weight)
                        root_histograms["Gen"]["Events"]["MesonType_BG"].Fill(genTauID, weight)
                        root_histograms["Reco"]["Events"]["RecoMesonType_BG"].Fill(recoTauID, weight)
                        root_histograms["Reco"]["Events"]["Omega_BG"].Fill(w, weight)
                        root_histograms["Reco"]["Events"]["CosTheta_BG"].Fill(cos_theta, weight)
                        root_histograms["Reco"]["Events"]["CosPsi_BG"].Fill(cos_psi, weight)
                        root_histograms["Reco"]["Events"]["OmegaCosTheta_BG"].Fill(w, cos_theta_rho, weight)
                        root_histograms["Gen"]["Events"]["OmegaCosThetaTau_GEN_BG"].Fill(gen_w, gen_cos_theta_tau, weight)
                        root_histograms["Gen"]["Events"]["Omega_GEN_BG"].Fill(gen_w, weight)
                        root_histograms["Gen"]["Events"]["CosTheta_GEN_BG"].Fill(gen_cos_theta, weight)
                        root_histograms["Gen"]["Events"]["CosPsi_GEN_BG"].Fill(gen_cos_psi, weight)

                        if genTauID == -13:
                            root_histograms["Reco"]["Events"]["RecoMesonEOverBeamE_BGMuon"].Fill(recoMesonP4.E() / beamE, weight)
                            root_histograms["Matched"]["Events"]["MesonEOverBeamE_BGMuon"].Fill(genMesonP4.E() / beamE, weight)
                            root_histograms["Reco"]["Events"]["RecoMesonE_BGMuon"].Fill(recoMesonP4.E(), weight)
                            root_histograms["Reco"]["Events"]["RecoMesonCosTheta_BGMuon"].Fill(math.cos(recoMesonP4.Theta()), weight)
                            root_histograms["Matched"]["Events"]["MesonCosTheta_BGMuon"].Fill(math.cos(genMesonP4.Theta()), weight)
                            root_histograms["Reco"]["Events"]["RecoMeson_BGMuon_PhiTheta"].Fill(recoMesonP4.Theta(), recoMesonP4.Phi(), weight)
                            root_histograms["Reco"]["Events"]["RecoMeson_BGMuon_PtTheta"].Fill(recoMesonP4.Theta(), recoMesonP4.Pt(), weight)
                            root_histograms["Reco"]["Events"]["Omega_BGMuon"].Fill(w, weight)
                            root_histograms["Reco"]["Events"]["OmegaCosTheta_BGMuon"].Fill(w, cos_theta_rho, weight)
                            root_histograms["Reco"]["Events"]["CosTheta_BGMuon"].Fill(cos_theta, weight)
                            root_histograms["Reco"]["Events"]["CosPsi_BGMuon"].Fill(cos_psi, weight)
                            root_histograms["Gen"]["Events"]["OmegaCosThetaTau_GEN_BGMuon"].Fill(gen_w, gen_cos_theta_tau, weight)
                            root_histograms["Gen"]["Events"]["Omega_GEN_BGMuon"].Fill(gen_w, weight)
                            root_histograms["Gen"]["Events"]["CosTheta_GEN_BGMuon"].Fill(gen_cos_theta, weight)
                            root_histograms["Gen"]["Events"]["CosPsi_GEN_BGMuon"].Fill(gen_cos_psi, weight)
                        elif genTauID == -11:
                            root_histograms["Reco"]["Events"]["RecoMesonEOverBeamE_BGEle"].Fill(recoMesonP4.E() / beamE, weight)
                            root_histograms["Matched"]["Events"]["MesonEOverBeamE_BGEle"].Fill(genMesonP4.E() / beamE, weight)
                            root_histograms["Reco"]["Events"]["RecoMesonCosTheta_BGEle"].Fill(math.cos(recoMesonP4.Theta()), weight)
                            root_histograms["Matched"]["Events"]["MesonCosTheta_BGEle"].Fill(math.cos(genMesonP4.Theta()), weight)
                            root_histograms["Reco"]["Events"]["RecoMeson_BGEle_PhiTheta"].Fill(recoMesonP4.Theta(), recoMesonP4.Phi(), weight)
                            root_histograms["Reco"]["Events"]["RecoMeson_BGEle_PtTheta"].Fill(recoMesonP4.Theta(), recoMesonP4.Pt(), weight)
                            root_histograms["Reco"]["Events"]["Omega_BGEle"].Fill(w, weight)
                            root_histograms["Reco"]["Events"]["OmegaCosTheta_BGEle"].Fill(w, cos_theta_rho, weight)
                            root_histograms["Reco"]["Events"]["CosTheta_BGEle"].Fill(cos_theta, weight)
                            root_histograms["Reco"]["Events"]["CosPsi_BGEle"].Fill(cos_psi, weight)
                            root_histograms["Gen"]["Events"]["OmegaCosThetaTau_GEN_BGEle"].Fill(gen_w, gen_cos_theta_tau, weight)
                            root_histograms["Gen"]["Events"]["Omega_GEN_BGEle"].Fill(gen_w, weight)
                            root_histograms["Gen"]["Events"]["CosTheta_GEN_BGEle"].Fill(gen_cos_theta, weight)
                            root_histograms["Gen"]["Events"]["CosPsi_GEN_BGEle"].Fill(gen_cos_psi, weight)
                        elif genTauID == 0:
                            root_histograms["Reco"]["Events"]["RecoMesonEOverBeamE_BGPion"].Fill(recoMesonP4.E() / beamE, weight)
                            root_histograms["Matched"]["Events"]["MesonEOverBeamE_BGPion"].Fill(genMesonP4.E() / beamE, weight)
                            root_histograms["Reco"]["Events"]["RecoMesonCosTheta_BGPion"].Fill(math.cos(recoMesonP4.Theta()), weight)
                            root_histograms["Matched"]["Events"]["MesonCosTheta_BGPion"].Fill(math.cos(genMesonP4.Theta()), weight)
                            root_histograms["Reco"]["Events"]["Omega_BGPion"].Fill(w, weight)
                            root_histograms["Reco"]["Events"]["OmegaCosTheta_BGPion"].Fill(w, cos_theta_rho, weight)
                            root_histograms["Reco"]["Events"]["CosTheta_BGPion"].Fill(cos_theta, weight)
                            root_histograms["Reco"]["Events"]["CosPsi_BGPion"].Fill(cos_psi, weight)
                            root_histograms["Gen"]["Events"]["OmegaCosThetaTau_GEN_BGPion"].Fill(gen_w, gen_cos_theta_tau, weight)
                            root_histograms["Gen"]["Events"]["Omega_GEN_BGPion"].Fill(gen_w, weight)
                            root_histograms["Gen"]["Events"]["CosTheta_GEN_BGPion"].Fill(gen_cos_theta, weight)
                            root_histograms["Gen"]["Events"]["CosPsi_GEN_BGPion"].Fill(gen_cos_psi, weight)
                        elif genTauID == 1:
                            root_histograms["Reco"]["Events"]["RecoMesonEOverBeamE_BGRho"].Fill(recoMesonP4.E() / beamE, weight)
                            root_histograms["Matched"]["Events"]["MesonEOverBeamE_BGRho"].Fill(genMesonP4.E() / beamE, weight)
                            root_histograms["Reco"]["Events"]["RecoMesonCosTheta_BGRho"].Fill(math.cos(recoMesonP4.Theta()), weight)
                            root_histograms["Matched"]["Events"]["MesonCosTheta_BGRho"].Fill(math.cos(genMesonP4.Theta()), weight)
                            root_histograms["Reco"]["Events"]["Omega_BGRho"].Fill(w, weight)
                            root_histograms["Reco"]["Events"]["OmegaCosTheta_BGRho"].Fill(w, cos_theta_rho, weight)
                            root_histograms["Reco"]["Events"]["CosTheta_BGRho"].Fill(cos_theta, weight)
                            root_histograms["Reco"]["Events"]["CosPsi_BGRho"].Fill(cos_psi, weight)
                            root_histograms["Gen"]["Events"]["OmegaCosThetaTau_GEN_BGRho"].Fill(gen_w, gen_cos_theta_tau, weight)
                            root_histograms["Gen"]["Events"]["Omega_GEN_BGRho"].Fill(gen_w, weight)
                            root_histograms["Gen"]["Events"]["CosTheta_GEN_BGRho"].Fill(gen_cos_theta, weight)
                            root_histograms["Gen"]["Events"]["CosPsi_GEN_BGRho"].Fill(gen_cos_psi, weight)
                        elif genTauID == 10:
                            root_histograms["Reco"]["Events"]["RecoMesonEOverBeamE_BGA1"].Fill(recoMesonP4.E() / beamE, weight)
                            root_histograms["Matched"]["Events"]["MesonEOverBeamE_BGA1"].Fill(genMesonP4.E() / beamE, weight)
                            root_histograms["Reco"]["Events"]["RecoMesonCosTheta_BGA1"].Fill(math.cos(recoMesonP4.Theta()), weight)
                            root_histograms["Matched"]["Events"]["MesonCosTheta_BGA1"].Fill(math.cos(genMesonP4.Theta()), weight)
                            root_histograms["Reco"]["Events"]["Omega_BGA1"].Fill(w, weight)
                            root_histograms["Reco"]["Events"]["OmegaCosTheta_BGA1"].Fill(w, cos_theta_rho, weight)
                            root_histograms["Reco"]["Events"]["CosTheta_BGA1"].Fill(cos_theta, weight)
                            root_histograms["Reco"]["Events"]["CosPsi_BGA1"].Fill(cos_psi, weight)
                            root_histograms["Gen"]["Events"]["OmegaCosThetaTau_GEN_BGA1"].Fill(gen_w, gen_cos_theta_tau, weight)
                            root_histograms["Gen"]["Events"]["Omega_GEN_BGA1"].Fill(gen_w, weight)
                            root_histograms["Gen"]["Events"]["CosTheta_GEN_BGA1"].Fill(gen_cos_theta, weight)
                            root_histograms["Gen"]["Events"]["CosPsi_GEN_BGA1"].Fill(gen_cos_psi, weight)
                        else:
                            root_histograms["Reco"]["Events"]["RecoMesonEOverBeamE_BGOther"].Fill(recoMesonP4.E() / beamE, weight)
                            root_histograms["Matched"]["Events"]["MesonEOverBeamE_BGOther"].Fill(genMesonP4.E() / beamE, weight)
                            root_histograms["Reco"]["Events"]["RecoMesonCosTheta_BGOther"].Fill(math.cos(recoMesonP4.Theta()), weight)
                            root_histograms["Matched"]["Events"]["MesonCosTheta_BGOther"].Fill(math.cos(genMesonP4.Theta()), weight)
                            root_histograms["Reco"]["Events"]["Omega_BGOther"].Fill(w, weight)
                            root_histograms["Reco"]["Events"]["OmegaCosTheta_BGOther"].Fill(w, cos_theta_rho, weight)
                            root_histograms["Reco"]["Events"]["CosTheta_BGOther"].Fill(cos_theta, weight)
                            root_histograms["Reco"]["Events"]["CosPsi_BGOther"].Fill(cos_psi, weight)
                            root_histograms["Gen"]["Events"]["OmegaCosThetaTau_GEN_BGOther"].Fill(gen_w, gen_cos_theta_tau, weight)
                            root_histograms["Gen"]["Events"]["Omega_GEN_BGOther"].Fill(gen_w, weight)
                            root_histograms["Gen"]["Events"]["CosTheta_GEN_BGOther"].Fill(gen_cos_theta, weight)
                            root_histograms["Gen"]["Events"]["CosPsi_GEN_BGOther"].Fill(gen_cos_psi, weight)

                else:  # no gen taus matched
                    root_histograms["Reco"]["Events"]["RecoMesonEOverBeamE_BG"].Fill(recoMesonP4.E() / beamE, weight)
                    root_histograms["Reco"]["Events"]["RecoMesonCosTheta_BG"].Fill(math.cos(recoMesonP4.Theta()), weight)
                    root_histograms["Matched"]["Events"]["MesonEOverBeamE_BG"].Fill(0, weight)
                    root_histograms["Matched"]["Events"]["MesonType_BG"].Fill(-3, weight)
                    root_histograms["Reco"]["Events"]["RecoMesonType_BG"].Fill(recoTauID, weight)
                    root_histograms["Reco"]["Events"]["Omega_BG"].Fill(w, weight)
                    root_histograms["Reco"]["Events"]["OmegaCosTheta_BG"].Fill(w, cos_theta_rho, weight)
                    root_histograms["Gen"]["Events"]["OmegaCosThetaTau_GEN_BG"].Fill(gen_w, gen_cos_theta_tau, weight)
                    root_histograms["Reco"]["Events"]["CosTheta_BG"].Fill(cos_theta, weight)
                    root_histograms["Reco"]["Events"]["CosPsi_BG"].Fill(cos_psi, weight)
                    root_histograms["Gen"]["Events"]["Omega_GEN_BG"].Fill(gen_w, weight)
                    root_histograms["Gen"]["Events"]["CosTheta_GEN_BG"].Fill(gen_cos_theta, weight)
                    root_histograms["Gen"]["Events"]["CosPsi_GEN_BG"].Fill(gen_cos_psi, weight)

            eventid += 1

    if skipped_files:
        logger_io.warning("Worker %d: %d file(s) skipped due to read errors",
                          worker_id, len(skipped_files))

    # ── Escribir fichero parcial ──────────────────────────────────────────────
    partial_outfile.cd()
    for tree_key in trees:
        write_histograms_recursive(root_histograms_super[tree_key])
        trees[tree_key].Write()
    partial_outfile.Close()

    logger_io.info("Worker %d: terminado. Events=%d Selected=%d",
                   worker_id, totalEvents, selectedEvents)
    return partial_path, totalEvents, selectedEvents, sumWeights, sumWeightsP1, sumWeightsM1


# ── Merge ─────────────────────────────────────────────────────────────────────

def merge_partial_root_files(partial_files, final_output):
    """Fusiona ficheros ROOT parciales con hadd; fallback a TFileMerger."""
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
    parser.add_argument("--sys-err", type=str, default="config/systematics/err_sys.yml",
                        help="YAML file with systematics errors to apply")
    parser.add_argument("--test-extremes", action="store_true",
                        help="Test the extremes of the photon energy resolution")
    parser.add_argument("--sin-eff", type=float, default=None,
                        help="Effective sin^2 theta_W to use in the weights calculation")
    parser.add_argument("--n-workers", type=int, default=None,
                        help="Número de procesos paralelos (default: min(n_files, n_cpus))")
    parser.add_argument("-e", "--electron-cut", type=float, default=10,
                        help="Minimum electron energy to consider (GeV)")
    parser.add_argument("-u", "--muon-cut", type=float, default=10,
                        help="Minimum muon energy to consider (GeV)")

def main():
    # 1. Cargar configuración (solo YAML, sin objetos ROOT)
    general_configs = myutils.setup_analysis_config(_DEFAULT_CONFIG, _OUTPUT_BASE,
                                                    parser_hook=my_hook)
    loggers    = general_configs["loggers"]
    run_config = general_configs["config"]
    args       = general_configs["args"]

    dRMax         = run_config["cuts"]["dRMax"]
    tauPCut       = run_config["cuts"]["tauCut"]
    minPTauPhoton = run_config["cuts"]["TauPhotonPCut"]
    minPTauPion   = run_config["cuts"]["TauPionPCut"]
    PNeutron      = run_config["cuts"]["NeutronCut"]
    dRMatch       = run_config["cuts"]["MatchedGenMinDR"]
    generalPCut   = run_config["cuts"]["generalPCut"]
    selectDecay   = general_configs["decay"]
    outputpath    = general_configs["outputpath"]
    fileOutName   = os.path.join(outputpath, general_configs["fileOutName"])

    sys_errors    = run_config.get("systematics_errors", {})
    photon_config = sys_errors.get("photon_config", {})
    test_extremes = args.test_extremes
    sample        = run_config["general"]["sample"]
    test_arg      = general_configs["flags"]["test"]
    gen_taus_sample = general_configs.get("has_gen_taus", False)

    gatr_results_path = args.gatr_result
    histogram_config  = general_configs.get("histograms_config", {})

    # Extraer el stem del nombre de fichero de salida (sin extensión ni directorio)
    fileOutName_base = Path(fileOutName).stem

    # 2. Obtener lista de ficheros (sin validación ROOT para no crear TFile antes del fork)
    filenames, mlpf_results = myutils.get_root_trees_path(
        sample, gatr_results_path, loggers, test_arg, args,
        skip_root_validation=True,
    )
    loggers["io"].info("Found %d input files", len(filenames))
    if not filenames:
        loggers["io"].error("No input files found. Aborting.")
        sys.exit(1)

    # 3. Determinar número de workers
    if args.input_list and len(args.input_list) == 1:
        n_workers = 1
    else:
        n_workers = args.n_workers or min(len(filenames), os.cpu_count() or 1)
    loggers["io"].info("Using %d workers for %d files", n_workers, len(filenames))

    # 4. Modo secuencial: n_workers==1
    if n_workers == 1:
        loggers["io"].info("Sequential mode (n_workers=1).")
        config_bundle = {
            "dRMax": dRMax, "tauPCut": tauPCut, "minPTauPhoton": minPTauPhoton,
            "minPTauPion": minPTauPion, "PNeutron": PNeutron, "generalPCut": generalPCut,
            "photon_config": photon_config, "test_extremes": test_extremes,
            "selectDecay": selectDecay, "sample": sample,
            "gatr_results_path": gatr_results_path, "sin_eff": args.sin_eff,
            "test_pfo": args.test_pfo, "outputpath": outputpath,
            "fileOutName_base": fileOutName_base, "histogram_config": histogram_config,
            "gen_taus_sample": gen_taus_sample,
        }
        partial_path, tot, sel, sw, swp1, swm1 = process_chunk_stage1(
            filenames, mlpf_results, 0, config_bundle, 0)
        # El modo secuencial ya escribe el partial → lo renombramos
        import shutil
        shutil.move(partial_path, fileOutName)
        loggers["io"].info("Sequential done: events=%d selected=%d", tot, sel)
    else:
        # 5. Modo paralelo
        file_chunks  = split_filenames(filenames, n_workers)
        mlpf_chunks  = split_mlpf(mlpf_results, file_chunks)
        n_chunks     = len(file_chunks)
        event_offsets = [sum(len(file_chunks[j]) for j in range(i)) * 1000
                         for i in range(n_chunks)]

        config_bundle = {
            "dRMax": dRMax, "tauPCut": tauPCut, "minPTauPhoton": minPTauPhoton,
            "minPTauPion": minPTauPion, "PNeutron": PNeutron, "generalPCut": generalPCut,
            "photon_config": photon_config, "test_extremes": test_extremes,
            "selectDecay": selectDecay, "sample": sample,
            "gatr_results_path": gatr_results_path, "sin_eff": args.sin_eff,
            "test_pfo": args.test_pfo, "outputpath": outputpath,
            "fileOutName_base": fileOutName_base, "histogram_config": histogram_config,
            "gen_taus_sample": gen_taus_sample, "minPTauElectron": args.electron_cut, "minPTauMuon": args.muon_cut,
        }

        # Fork ANTES de que el proceso principal abra cualquier TFile
        ctx = multiprocessing.get_context("fork")
        partial_files = []
        totals = {"events": 0, "selected": 0, "sw": 0.0, "swp1": 0.0, "swm1": 0.0}
        t_start = time.time()

        loggers["io"].info("Launching %d workers...", n_chunks)
        with ProcessPoolExecutor(max_workers=n_chunks, mp_context=ctx) as executor:
            futures = {
                executor.submit(
                    process_chunk_stage1,
                    file_chunks[i], mlpf_chunks[i], event_offsets[i],
                    config_bundle, i,
                ): i
                for i in range(n_chunks)
            }
            for n_done, future in enumerate(as_completed(futures), start=1):
                wid = futures[future]
                try:
                    partial_path, tot, sel, sw, swp1, swm1 = future.result()
                    partial_files.append(partial_path)
                    totals["events"]   += tot
                    totals["selected"] += sel
                    totals["sw"]       += sw
                    totals["swp1"]     += swp1
                    totals["swm1"]     += swm1
                    elapsed = time.time() - t_start
                    print(f"  [{n_done}/{n_chunks}] worker {wid} terminado "
                          f"({tot} eventos, {elapsed:.1f}s acumulado)", flush=True)
                except Exception as exc:
                    loggers["io"].error("Worker %d falló: %s", wid, exc)
                    raise

        # 6. Merge
        loggers["io"].info("Merging %d partial files → %s", len(partial_files), fileOutName)
        merge_partial_root_files(partial_files, fileOutName)

        # 7. Limpiar parciales
        for p in partial_files:
            try:
                os.remove(p)
            except OSError:
                pass

        loggers["io"].info("Total events=%d selected=%d (%.1fs)",
                           totals["events"], totals["selected"], time.time() - t_start)

    # 8. Guardar resumen CSV y config
    decay_str = general_configs["decay"]
    results_dict = {
        "TotalEvents":   totals.get("events", 0) if n_workers > 1 else tot,
        "SelectedEvents": totals.get("selected", 0) if n_workers > 1 else sel,
        "SumWeights":    totals.get("sw", 0.0)   if n_workers > 1 else sw,
        "SumWeightsP1":  totals.get("swp1", 0.0) if n_workers > 1 else swp1,
        "SumWeightsM1":  totals.get("swm1", 0.0) if n_workers > 1 else swm1,
    }
    results_df = pd.DataFrame(results_dict, index=[0])
    results_df.to_csv(os.path.join(outputpath, f"results_summary_{decay_str}.csv"), index=False)

    
    output_config_file = os.path.join(outputpath, "config.yaml")
    # add args to run_config for completeness
    run_config["args"] = vars(args)
    run_config["run_Type"]="analysisRHOTree"
    with open(output_config_file, "w") as f:
        yaml.dump(run_config, f)
        
    loggers["io"].info("Config guardado en %s", output_config_file)
    loggers["io"].info("Fichero de salida: %s", fileOutName)


if __name__ == "__main__":
    main()
