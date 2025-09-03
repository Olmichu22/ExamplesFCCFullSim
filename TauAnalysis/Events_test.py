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

from modules import ParticleObjects, electronReco, muonReco, myutils, pi0Reco, tauReco
from modules.ParticleObjects import RecoParticle


def write_histograms_recursive(obj):
    """
    Recorre un diccionario anidado y ejecuta `.Write()` en cada objeto tipo ROOT histogram.
    """
    if isinstance(obj, dict):
        for value in obj.values():
            write_histograms_recursive(value)
    else:
        # Si no es diccionario, asumimos que es un histograma de ROOT
        try:
            obj.Write()
        except AttributeError:
            print(f"Objeto {obj} no tiene método .Write(). Ignorado.")


# ----------------------------------------------------------------------------
# Load config (necessary for set up the logger)
default_config = "config/default/taurecolong.yaml"
# Output Configuration
outputbasepath = "Tests/TauReco/"

general_configs = myutils.setup_analysis_config(default_config, outputbasepath)


loggers = general_configs["loggers"]

run_config = general_configs["config"]

# config = myutils.load_yaml_config(args.config, default_config)


# Cut Configuration
dRMax = run_config["cuts"]["dRMax"]
minPTauPhoton = run_config["cuts"]["TauPhotonPCut"]
minPTauPion = run_config["cuts"]["TauPionPCut"]
PNeutron = run_config["cuts"]["NeutronCut"]
dRMatch = run_config["cuts"]["MatchedGenMinDR"]
generalPCut = run_config["cuts"]["generalPCut"]

selectDecay = general_configs["decay"]

outputpath = general_configs["outputpath"]
fileOutName = os.path.join(
    general_configs["outputpath"], general_configs["fileOutName"]
)

logger_config = loggers["config"]
logger_io = loggers["io"]
logger_process = loggers["processing"]
logger_pi0mass = loggers["pi0mass"]


# Continue with the rest of configs

# ------------------------------------------------------------------------
# General Configuration
sample = run_config["general"]["sample"]
matched_cm_arg = general_configs["flags"]["matched_cm"]
test_arg = general_configs["flags"]["test"]

logger_config.info("Configuration loaded!")
logger_config.info("Configuration:\n%s", pprint.pformat(general_configs, indent=4))


# ------------------------------------------------------------------------
gatr_results_path = general_configs["args"].gatr_result

filenames, mlpf_results = myutils.get_root_trees_path(
    sample, gatr_results_path, loggers, test_arg
)
reader = root_io.Reader(filenames)
logger_io.info("Read %d files", len(filenames))
logger_io.info("First %s files.", filenames[:10])

# Configs and reading finished
# ----------------------------------------------------------------------


# collections to use
genparts = "MCParticles"
pfobjects = "PandoraPFOs"
# pfobjects ="TightSelectedPandoraPFOs"


for eventid, event in enumerate(reader.get("events")):
    # if gatr_results_path is not None and eventid > len(gatr_results) - 1:
    #     logger_process.info("Reached the end of GATr results, stopping processing.")
    #     break
    logger_process.debug("Processing event %d", eventid)
    # if countEvents % 1000 == 0:
    #     logger_process.info("Processing event %d", countEvents)
    # countEvents += 1

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
        particles = mlpf_results.get(eventid, {})
        recoTau = tauReco.findAllTaus(
            particles,
            dRMax,
            minPTauPhoton,
            minPTauPion,
            PNeutron,
            generalPCut,
            charge_condition=False,
        )
        recoElectrons = electronReco.findAllElectrons(particles, generalPCut)
        recoMuons = muonReco.findAllMuons(particles, generalPCut)
    else:
        recoTau = tauReco.findAllTaus(
            pfos, dRMax, minPTauPhoton, minPTauPion, PNeutron, generalPCut
        )
        recoElectrons = electronReco.findAllElectrons(pfos, generalPCut)
        recoMuons = muonReco.findAllMuons(pfos, generalPCut)

    nRecoTaus = len(recoTau)
    nRecoElectrons = len(recoElectrons)
    nRecoMuons = len(recoMuons)

    recoTaus = {}
    pidx = 0
    for taui in range(nRecoTaus):
        recoTaus[pidx] = recoTau[taui]
        pidx += 1
    for elei in range(nRecoElectrons):
        recoTaus[pidx] = recoElectrons[elei]
        pidx += 1
    for mui in range(nRecoMuons):
        recoTaus[pidx] = recoMuons[mui]
        pidx += 1
    nRecoTaus = len(recoTaus)

    logger_process.debug(
        "Found %d reconstructed taus. Details:\n%s",
        nRecoTaus,
        "\n".join("RecoTau %d: %s" % (i, tau) for i, tau in recoTaus.items()),
    )

    for i in range(0, nGenTaus):
        genVisTauP4 = genTaus[i].getvisMomentum()
        genTauId = genTaus[i].getID()
        genTauQ = genTaus[i].getCharge()
        genTauP4 = genTaus[i].getMomentum()
        genTauDR = genTaus[
            i
        ].getMaxAngle()  # Maximum angle between the tau and its constituents
        genTauNConsts = genTaus[i].getnConst()
        genTauConsts = genTaus[i].getDaughters()

        for const in genTauConsts:
            for c in range(0, genTauNConsts):
                const = genTauConsts[c]
                dauP4 = ROOT.TLorentzVector()
                try:
                    dauP4.SetXYZM(
                        const.getMomentum().x,
                        const.getMomentum().y,
                        const.getMomentum().z,
                        const.getMass(),
                    )
                except AttributeError:
                    # If the dau does not have a mass, we assume it is a photon
                    dauP4.SetXYZM(
                        const.getMomentum().X(),
                        const.getMomentum().Y(),
                        const.getMomentum().Z(),
                        const.getMass(),
                    )
                if genTauId == 0:
                    cum_p4 += dauP4
                # Si se trata de un pion neutro, lo añadimos a la suma de fotones
                if const.getPDG() == 111:
                    photons = const.getDaughters()
                    for photon in photons:
                        print(photon.getPDG())
                        photonP4 = ROOT.TLorentzVector()
                        photonP4.SetXYZM(
                            photon.getMomentum().x,
                            photon.getMomentum().y,
                            photon.getMomentum().z,
                            photon.getMass(),
                        )
                    exit(0)