import sys, os, math
from array import array
import ROOT
from ROOT import TFile, TTree, TH1F, TH2F
import numpy as np
from podio import root_io
import edm4hep
from pathlib import Path
import pprint
import yaml
import pandas as pd
import pickle
from modules import ParticleObjects
from modules.ParticleObjects import RecoParticle

from modules import pi0Reco
from modules import tauReco
from modules import myutils

import logging


import argparse

parser = argparse.ArgumentParser(
    description="Configure the analysis",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)
parser.add_argument("-f", "--sample")
parser.add_argument("-o", "--outfile")
parser.add_argument("-d", "--decay", type=int)  # GEN
parser.add_argument(
    "-p", "--TauPhotonPCut", type=float
)  
parser.add_argument("-i", "--TauPionPCut", type=float)
parser.add_argument("-R", "--dRMax", type=float)
parser.add_argument("-n", "--NeutronCut", type=float)
parser.add_argument("-g", "--generalPCut", type=float)
parser.add_argument("-r", "--MatchedGenMinDR", type=float)
parser.add_argument(
    "-m",
    "--matchedCM",
    default="True",
    type=str,
    help="Use only matched taus to compute confusion matrix.",
)
parser.add_argument(
    "-t",
    "--test",
    type=str,
    help="Run in test mode with limited number of files",
)
parser.add_argument(
    "-c", "--config", type=str, help="Configuration file"
)
parser.add_argument(
    "-v",
    "--verbose",
    action="count",
    default=0,
    help="Increase verbosity level: -v for INFO, -vv for DEBUG",
)

parser.add_argument(
    "--gatr-result",
    type=str,
    help="Path to GATR result for the analysis.",
)

args = parser.parse_args()

# ----------------------------------------------------------------------------
# Load config (necessary for set up the logger)
default_config = "config/default/taurecolong.yaml"
config = myutils.load_yaml_config(args.config, default_config)


# Cut Configuration
config["cuts"]["dRMax"] = args.dRMax if args.dRMax != None else config["cuts"]["dRMax"]
config["cuts"]["TauPhotonPCut"] = (
    args.TauPhotonPCut
    if args.TauPhotonPCut != None
    else config["cuts"]["TauPhotonPCut"]
)
config["cuts"]["TauPionPCut"] = (
    args.TauPionPCut if args.TauPionPCut != None else config["cuts"]["TauPionPCut"]
)
config["cuts"]["NeutronCut"] = (
    args.NeutronCut if args.NeutronCut != None else config["cuts"]["NeutronCut"]
)
config["cuts"]["MatchedGenMinDR"] = (
    args.MatchedGenMinDR
    if args.MatchedGenMinDR != None
    else config["cuts"]["MatchedGenMinDR"]
)
config["cuts"]["generalPCut"] = (
    args.generalPCut if args.generalPCut != None else config["cuts"]["generalPCut"]
)
dRMax = config["cuts"]["dRMax"]
minPTauPhoton = config["cuts"]["TauPhotonPCut"]
minPTauPion = config["cuts"]["TauPionPCut"]
PNeutron = config["cuts"]["NeutronCut"]
dRMatch = config["cuts"]["MatchedGenMinDR"]
generalPCut = config["cuts"]["generalPCut"]

# We can use same config but different decay mode
# Priority is given to the decay mode in the command line
if args.decay not in config["general"]["decay"] and args.decay != None:
    config["general"]["decay"].append(args.decay)
    selectDecay = args.decay
else:
    selectDecay = args.decay if args.decay != None else config["general"]["decay"][0]

config["general"]["outfile"] = (
    args.outfile if args.outfile != None else config["general"]["outfile"]
)
outfile = config["general"]["outfile"]


# Output Configuration
outputbasepath = "Results/TauReco/"


cut_string = f"_{dRMax}_tph{minPTauPhoton}_tpi{minPTauPion}_n{PNeutron}_g{generalPCut}"
decayString = f"decay{selectDecay}" + cut_string
if selectDecay == -777:
    decayString = "decayAll" + cut_string
fileOutName = outfile + decayString + ".root"


outputpath = outputbasepath + outfile + cut_string[1:] + "/"
if args.gatr_result is not None:
    outputpath = "GATr_" + outputpath

config["output"]["outputpath"] = outputpath
# Check if config["output"]["outputfile"] is a list
if type(config["output"]["outputfile"]) is not list:
    if config["output"]["outputfile"] is None:
        config["output"]["outputfile"] = []
    else:
        config["output"]["outputfile"] = [config["output"]["outputfile"]]

if fileOutName not in config["output"]["outputfile"]:
    config["output"]["outputfile"].append(fileOutName)

if not os.path.exists(outputpath):
    os.makedirs(outputpath)

# Once set the output path, we can set the logger
if args.verbose == 0:
    log_level = logging.WARNING  # Only warnings and errors
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(outputpath + "/" + "app.log", mode="w"),
    ]
elif args.verbose == 1:
    log_level = logging.INFO  # Informational messages
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(outputpath + "/" + "app.log", mode="w"),
    ]
elif args.verbose == 2:
    log_level = logging.DEBUG  # Debug messages for -vv or higher
    # Crear handlers por separado para configurarlos individualmente
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setLevel(logging.INFO)  # Terminal solo muestra INFO o superior
    file_handler = logging.FileHandler(outputpath + "/" + "app.log", mode="w")
    file_handler.setLevel(logging.DEBUG)   # Archivo guarda TODO (DEBUG y superior)
    handlers=[stream_handler, file_handler]
elif args.verbose > 2:
    log_level = logging.DEBUG  # Debug messages for -vv or higher
    handlers=[
        logging.FileHandler(outputpath + "/" + "app.log", mode="w"),
    ]
    


logging.basicConfig(
    level=log_level,
    format="%(asctime)s, %(levelname)s, [%(name)s] - %(message)s",
    handlers=handlers)

logger_config = logging.getLogger("config")
logger_io = logging.getLogger("io")
logger_process = logging.getLogger("processing")
logger_pi0mass = logging.getLogger("pi0mass")


# Continue with the rest of configs


# General Configuration
config["general"]["sample"] = (
    args.sample if args.sample != None else config["general"]["sample"]
)
config["general"]["matchedCM"] = (
    args.matchedCM if args.matchedCM != None else config["general"]["matchedCM"]
)
config["general"]["test"] = (
    args.test if args.test != None else config["general"]["test"]
)

sample = config["general"]["sample"]
matched_cm_arg = config["general"]["matchedCM"]
matched_cm = True if matched_cm_arg == "True" else False
test_arg = config["general"]["test"]
test = True if test_arg == "True" else False

logger_config.info("Configuration loaded!")
logger_config.info("Configuration:\n%s", pprint.pformat(config, indent=4))


# get all the files

# GATr reading config (if provided)

gatr_results_path = args.gatr_result

if gatr_results_path is not None:
    if not os.path.exists(gatr_results_path):
        logger_io.error("GATr results path %s does not exist.", gatr_results_path)
        sys.exit(1)
    else:
        logger_io.info("Using GATr results from %s", gatr_results_path)

    with open(gatr_results_path, "rb") as f:
        gatr_results = pickle.load(f)
        logger_io.info("GATr results loaded successfully.")

    print(gatr_results)

# Simulation files
path = "/pnfs/ciemat.es/data/cms/store/user/cepeda/FCC/FullSim/"
file = "out_reco_edm4hep_edm4hep"
filenames = []
dir_path = path + "/" + sample

nfiles = len(os.listdir(dir_path))

nfiles = 1000
if test == True:
    nfiles = 2

if gatr_results_path is not None:
    print(len(gatr_results))
    nfiles = len(gatr_results)//1000

logger_io.info("Reading files from %s", dir_path)
for i in range(1, nfiles + 1):
    filename = dir_path + "/" + file + "_{}.root".format(i)
    logger_io.debug("Reading file %s", filename)
    my_file = Path(filename)
    if my_file.is_file():
        root_file = myutils.open_root_file(filename)
        if not root_file or root_file.IsZombie():
            logger_io.warning("File %s is a zombie or could not be opened.", filename)
            continue
        filenames.append(filename)

reader = root_io.Reader(filenames)

logger_io.info("Read %d files", len(filenames))
logger_io.info("First %s files.", filenames[:10]) 

# Configs and reading finished
# ----------------------------------------------------------------------


# collections to use
genparts = "MCParticles"
pfobjects = "PandoraPFOs"
# pfobjects ="TightSelectedPandoraPFOs"

# Defining many histogram
hGenTauPt = TH1F("histoGenTauPt", "", 250, 0, 50)
hGenVisTauPt = TH1F("histoGenTauVisPt", "", 250, 0, 50)
hGenTauP = TH1F("histoGenTauP", "", 250, 0, 50)
hGenVisTauP = TH1F("histoGenTauVisP", "", 250, 0, 50)
hGenTauType = TH1F("histoGenTauType", "", 21, -1, 20)
hGenVisTauMass = TH1F("histoGenTauVisMass", "", 500, 0, 10)
hGenTauQ = TH1F("histoGenTauQ", "", 3, -1.5, 1.5)
hGenTauEta = TH1F("histoGenTauEta", "", 100, -5, 5)
hGenTauTheta = TH1F("histoGenTauTheta", "", 100, 0, 3.15)
hGenTauDR = TH1F("histoGenTauDR", "Angle of Tau Constituents", 100, 0, 1)

hMatchedGenTauPt = TH1F("histoMatchedGenTauPt", "", 250, 0, 50)
hMatchedGenVisTauPt = TH1F("histoMatchedGenTauVisPt", "", 250, 0, 50)
hMatchedGenTauP = TH1F("histoMatchedGenTauP", "", 250, 0, 50)
hMatchedGenVisTauP = TH1F("histoMatchedGenTauVisP", "", 250, 0, 50)

hMatchedGenTauType = TH1F("histoMatchedGenTauType", "", 21, -1, 20)
hMatchedGenVisTauMass = TH1F("histoMatchedGenTauVisMass", "", 500, 0, 10)
hMatchedGenTauQ = TH1F("histoMatchedGenTauQ", "", 3, -1.5, 1.5)
hMatchedGenTauEta = TH1F("histoMatchedGenTauEta", "", 100, -5, 5)
hMatchedGenTauTheta = TH1F("histoMatchedGenTauTheta", "", 100, 0, 3.15)

hMatchedGenTauDR = TH1F("histoMatchedGenTauDR", "Angle of Tau Constituents", 100, 0, 1)

hRecoTauPt = TH1F("histoRecoTauPt", "", 250, 0, 50)
hRecoTauP = TH1F("histoRecoTauP", "", 250, 0, 50)

hRecoTauMass = TH1F("histoRecoTauMass", "", 500, 0, 10)
hRecoTauType = TH1F("histoRecoTauType", "", 21, -1, 20)
hRecoTauQ = TH1F("histoRecoTauQ", "", 3, -1.5, 1.5)
hRecoTauEta = TH1F("histoRecoTauEta", "", 100, -5, 5)
hRecoTauTheta = TH1F("histoRecoTauTheta", "", 100, 0, 3.15)
hRecoTauDR = TH1F("histoRecoTauDR", "Angle of Tau Constituents", 100, 0, 1)




hRecoConstPi0Mass = TH1F("hRecoConstPi0Mass", "", 100, 0, 0.5)
hRecoConstTwoPhotonAngDist = TH1F("hRecoConstTwoPhotonAngDist", "", 100, 0, 2)

hRecoConstPi0MassFromPhotonMasstr = TH1F("hRecoConstPi0MassFromPhotonMasstr", "", 100, 0, 0.5)
hRecoConstPi0MassFromPhotonDiststr = TH1F("hRecoConstPi0MassFromPhotonDiststr", "", 100, 0, 0.5)


hGenConstPi0Mass = TH1F("hGenConstPi0Mass", "", 100, 0, 0.5)
hMatchedGenConstPi0Mass = TH1F("hMatchedConstPi0Mass", "", 100, 0, 0.5)
h2DPi0MassOverNPhoton = TH2F("hRecoPi0MassOverNPhoton", "", 100, 0, 2, 20, 0, 20)

# Hist for reconstructed photons in a1 (pi pi0 pi0), rho (pi0 pi0), and pi cases
# 3 photon cases
hRecoConstlessPhotonPa1strMass = TH1F("hRecoConstlessPhotonPa1strMass", "", 100, 0, 50)
hRecoConstlessPhotonPa1strMassZoom = TH1F("hRecoConstlessPhotonPa1strMassZoom", "", 100, 0, 2)
hRecoConstlessPhotonPa1strDist = TH1F("hRecoConstlessPhotonPa1strDist", "", 100, 0, 50)

hRecoConstxtraPhotonPrhostrMass = TH1F("hRecoConstxtraPhotonPrhostrMass", "", 100, 0, 50)
hRecoConstxtraPhotonPrhostrMassZoom = TH1F("hRecoConstxtraPhotonPrhostrMassZoom", "", 100, 0, 2)
hRecoConstxtraPhotonPrhostrDist = TH1F("hRecoConstxtraPhotonPrhostrDist", "", 100, 0, 50)

# 1 photon cases
hRecoConstlessPhotonPrho = TH1F("hRecoConstlessPhotonPrho", "", 100, 0, 50)
hRecoConstxtraPhotonPi = TH1F("hRecoConstxtraPhotonPi", "", 100, 0, 50)
hRecoConstlessPhotonPrhoZoom = TH1F("hRecoConstlessPhotonPrhoZoom", "", 100, 0, 2)
hRecoConstxtraPhotonPiZoom = TH1F("hRecoConstxtraPhotonPiZoom", "", 100, 0, 2)


# Photon P in the 3 photon case
hRecoThreePhotonMatchOnestrMassP = TH1F("hRecoThreePhotonMatchOnestrMassP", "", 100, 0, 50)
hRecoThreePhotonMatchOnestrDistP = TH1F("hRecoThreePhotonMatchOnestrDistP", "", 100, 0, 50)
hRecoThreePhotonMatchTwostrMassP = TH1F("hRecoThreePhotonMatchTwostrMassP", "", 100, 0, 50)
hRecoThreePhotonMatchTwostrDistP = TH1F("hRecoThreePhotonMatchTwostrDistP", "", 100, 0, 50)
hRecoThreePhotonNoMatchstrMassP = TH1F("hRecoThreePhotonNoMatchstrMassP", "", 100, 0, 50)
hRecoThreePhotonNoMatchstrDistP = TH1F("hRecoThreePhotonNoMatchstrDistP", "", 100, 0, 50)


#  Rho decay Pi P
hRecoRhoPiDecayP = TH1F("hRecoRhoPiDecayP", "", 100, 0, 50)
hGenRhoPiDecayP = TH1F("hGenRhoPiDecayP", "", 100, 0, 50)
# Rho decay two photons case P
hRecoRhoTwoPhotonDecayP = TH1F("hRecoRhoTwoPhotonDecayP", "", 100, 0, 50)
hGenRhoTwoPhotonDecayP = TH1F("hGenRhoTwoPhotonDecayP", "", 100, 0, 50)
hRecoRhoTwoPhotonDecaySumP = TH1F("hRecoRhoTwoPhotonDecaySumP", "", 100, 0, 50)
hGenRhoTwoPhotonDecaySumP = TH1F("hGenRhoTwoPhotonDecaySumP", "", 100, 0, 50)
# Hist of pi P vs sum of photons P

h2DRecoRhoTwoPhotonDecayPiPhotonSumP = TH2F("h2DRecoRhoTwoPhotonDecayPiPhotonSumP", "", 100, 0, 50, 100, 0, 50)
h2DGenRhoTwoPhotonDecayPiPhotonSumP = TH2F("h2DGenRhoTwoPhotonDecayPiPhotonSumP", "", 100, 0, 50, 100, 0, 50)

# Rho False decay (one photon)
hRecoRhoOnePhotonDecayPiP = TH1F("hRecoRhoOnePhotonDecayPiP", "", 100, 0, 50)
hGenRhoOnePhotonDecayPiP = TH1F("hGenRhoOnePhotonDecayPiP", "", 100, 0, 50)
hRecoRhoOnePhotonDecayPhotonP = TH1F("hRecoRhoOnePhotonDecayPhotonP", "", 100, 0, 50)
hGenRhoOnePhotonDecayPhotonP = TH1F("hGenRhoOnePhotonDecayPhotonP", "", 100, 0, 50)
hGenRhoOnePhotonDecayPhotonSumP = TH1F("hGenRhoOnePhotonDecayPhotonSumP", "", 100, 0, 50)
# Angle between the photons at gen level
hGenRhoOnePhotonDecayPhotonAng = TH1F("hGenRhoOnePhotonDecayPhotonAng", "", 100, 0, 2)
# Hist of pi P vs photon P at reco level:
h2DRecoRhoOnePhotonDecayPiPhotonP = TH2F("h2DRecoRhoOnePhotonDecayPiPhotonP", "",100, 0, 50, 100, 0, 50)
# Hist of pi P vs photon P at gen level:
h2DGenRhoOnePhotonDecayPiPhotonSumP = TH2F("h2DGenRhoOnePhotonDecayPiPhotonSumP", "",100, 0, 50, 100, 0, 50)

h2DTauPt = TH2F("histo2DTauPt", "", 250, 0, 50, 250, 0, 50)
h2DTauP = TH2F("histo2DTauP", "", 250, 0, 50, 250, 0, 50)
h2DTauDR = TH2F("histo2DTauDR", "", 100, 0, 1, 100, 0, 1)
h2DTauMass = TH2F("histo2DTauMass", "", 500, 0, 10, 500, 0, 10)
h2DTauType = TH2F("histo2DTauType", "", 21, -1, 20, 21, -1, 20)
h2DTauQ = TH2F("histo2DTauQ", "", 4, -2, 2, 4, -2, 2)

hResTauPt = TH1F("histoResTauPt", "", 500, -1, 1)
hResTauP = TH1F("histoResTauP", "", 500, -1, 1)
hResTauMass = TH1F("histoResTauMass", "", 500, -1, 1)

hNTaus = TH1F("histoNTaus", "", 6, 0, 6)
hNGenTaus = TH1F("histoNGenTaus", "", 6, 0, 6)
hNTausType = TH1F("histoNTausType", "", 6, 0, 6)
hNGenTausType = TH1F("histoNGenTausType", "", 6, 0, 6)

hMatchedTausPRes = TH1F("histoMatchedTausPRes", "", 500, -1, 1)
hMatchedTausPtRes = TH1F("histoMatchedTausPtRes", "", 500, -1, 1)
hMatchedTausChargeRes = TH1F("histoMatchedTausChargeRes", "", 500, -1, 1)
hMatchedTausMaxAngleRes = TH1F("histoMatchedTausMaxAngleRes", "", 500, -1, 1)
hMatchedTausNCompRes = TH1F("histoMatchedTausNCompRes", "", 500, -1, 1)

true_predicted_label = {"GenID": [], "True": [], "Predicted": [], "PhotonPredicted": []}
unmatched_true_label = {}
countEvents = 0
# run over all events

# Results plots
# Theta
hGenTauTheta0 = TH1F("histoGenTauTheta0", "", 100, 0, 3.15)
hGenTauTheta1 = TH1F("histoGenTauTheta1", "", 100, 0, 3.15)
hGenTauTheta2 = TH1F("histoGenTauTheta2", "", 100, 0, 3.15)
hGenTauTheta10 = TH1F("histoGenTauTheta10", "", 100, 0, 3.15)
hMatchedGenTauTheta0 = TH1F("histoMatchedGenTauTheta0", "", 100, 0, 3.15)
hMatchedGenTauTheta1 = TH1F("histoMatchedGenTauTheta1", "", 100, 0, 3.15)
hMatchedGenTauTheta2 = TH1F("histoMatchedGenTauTheta2", "", 100, 0, 3.15)
hMatchedGenTauTheta10 = TH1F("histoMatchedGenTauTheta10", "", 100, 0, 3.15)
hRecoTauTheta0 = TH1F("histoRecoTauTheta0", "", 100, 0, 3.15)
hRecoTauTheta1 = TH1F("histoRecoTauTheta1", "", 100, 0, 3.15)
hRecoTauTheta2 = TH1F("histoRecoTauTheta2", "", 100, 0, 3.15)
hRecoTauTheta10 = TH1F("histoRecoTauTheta10", "", 100, 0, 3.15)

hTauThetaRes0 = TH1F("histoTauThetaRes0", "", 500, -1, 1)
hTauThetaRes1 = TH1F("histoTauThetaRes1", "", 500, -1, 1)
hTauThetaRes2 = TH1F("histoTauThetaRes2", "", 500, -1, 1)
hTauThetaRes10 = TH1F("histoTauThetaRes10", "", 500, -1, 1)

hMatchedTauThetaRes0 = TH1F("histoMatchedTauThetaRes0", "", 500, -1, 1)
hMatchedTauThetaRes1 = TH1F("histoMatchedTauThetaRes1", "", 500, -1, 1)
hMatchedTauThetaRes2 = TH1F("histoMatchedTauThetaRes2", "", 500, -1, 1)
hMatchedTauThetaRes10 = TH1F("histoMatchedTauThetaRes10", "", 500, -1, 1)

# P (Momentum)
hGenTauP0 = TH1F("histoGenTauP0", "", 100, 0, 50)
hGenTauP1 = TH1F("histoGenTauP1", "", 100, 0, 50)
hGenTauP2 = TH1F("histoGenTauP2", "", 100, 0, 50)
hGenTauP10 = TH1F("histoGenTauP10", "", 100, 0, 50)
hGenTauVisP0 = TH1F("histoGenTauVisP0", "", 100, 0, 50)
hGenTauVisP1 = TH1F("histoGenTauVisP1", "", 100, 0, 50)
hGenTauVisP2 = TH1F("histoGenTauVisP2", "", 100, 0, 50)
hGenTauVisP10 = TH1F("histoGenTauVisP10", "", 100, 0, 50)
hMatchedGenTauP0 = TH1F("histoMatchedGenTauP0", "", 100, 0, 50)
hMatchedGenTauP1 = TH1F("histoMatchedGenTauP1", "", 100, 0, 50)
hMatchedGenTauP2 = TH1F("histoMatchedGenTauP2", "", 100, 0, 50)
hMatchedGenTauP10 = TH1F("histoMatchedGenTauP10", "", 100, 0, 50)
hMatchedGenTauVisP0 = TH1F("histoMatchedGenTauVisP0", "", 100, 0, 50)
hMatchedGenTauVisP1 = TH1F("histoMatchedGenTauVisP1", "", 100, 0, 50)
hMatchedGenTauVisP2 = TH1F("histoMatchedGenTauVisP2", "", 100, 0, 50)
hMatchedGenTauVisP10 = TH1F("histoMatchedGenTauVisP10", "", 100, 0, 50)

hRecoTauP0 = TH1F("histoRecoTauP0", "", 100, 0, 50)
hRecoTauP1 = TH1F("histoRecoTauP1", "", 100, 0, 50)
hRecoTauP2 = TH1F("histoRecoTauP2", "", 100, 0, 50)
hRecoTauP10 = TH1F("histoRecoTauP10", "", 100, 0, 50)

# Resolution P
hTauPRes0 = TH1F("histoTauPRes0", "", 500, -1, 1)
hTauPRes1 = TH1F("histoTauPRes1", "", 500, -1, 1)
hTauPRes2 = TH1F("histoTauPRes2", "", 500, -1, 1)
hTauPRes10 = TH1F("histoTauPRes10", "", 500, -1, 1)

hMatchedTauPRes0 = TH1F("histoMatchedTauPRes0", "", 500, -1, 1)
hMatchedTauPRes1 = TH1F("histoMatchedTauPRes1", "", 500, -1, 1)
hMatchedTauPRes2 = TH1F("histoMatchedTauPRes2", "", 500, -1, 1)
hMatchedTauPRes10 = TH1F("histoMatchedTauPRes10", "", 500, -1, 1)

result_labels = {}
result_labels["tau1"] = []
result_labels["tau2"] = []
result_labels["id-tau1"] = []
result_labels["id-tau2"] = []

for eventid, event in enumerate(reader.get("events")):
    if gatr_results_path is not None and eventid > len(gatr_results) - 1:
        logger_process.info("Reached the end of GATr results, stopping processing.")
        break
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
    if gatr_results_path is not None:
        recoTaus = tauReco.findAllTaus(
            gatr_results[f"event_{eventid}"], dRMax, minPTauPhoton, minPTauPion, PNeutron, generalPCut
        )
    else:
        recoTaus = tauReco.findAllTaus(
            pfos, dRMax, minPTauPhoton, minPTauPion, PNeutron, generalPCut
        )
        
    nRecoTaus = len(recoTaus)

    logger_process.debug(
        "Found %d reconstructed taus. Details:\n%s",
        nRecoTaus,
        "\n".join("RecoTau %d: %s" % (i, tau) for i, tau in recoTaus.items()),
    )

    foundGen = False

    nGenTausType = 0
    nTausType = 0
    nGenTausHad = 0
    
    if nGenTaus == 0:
        result_labels["tau1"].append(-999)
        result_labels["tau2"].append(-999)
        result_labels["id-tau1"].append(-999)
        result_labels["id-tau2"].append(-999)
    
    for i in range(0, nGenTaus):
        genVisTauP4 = genTaus[
            i
        ].getvisMomentum()  # to do: find a clearer dictionary for this
        genTauId = genTaus[i].getID()
        genTauQ = genTaus[i].getCharge()
        genTauP4 = genTaus[i].getMomentum()
        genTauDR = genTaus[
            i
        ].getMaxAngle()  # Maximum angle between the tau and its constituents
        genTauNConsts = genTaus[i].getnConst()
        genTauConsts = genTaus[i].getDaughters()

        if nGenTaus > 2 and i <=1:
            result_labels[f"tau{i+1}"].append(-999)
        elif nGenTaus <= 2:
            result_labels[f"tau{i+1}"].append(genTauId)
        
        # remove leptonic decays
        if genTauId >= 0:
            nGenTausHad += 1

        # if genTauId < 0:
        #    continue

        # nGenTausHad+=1

        # pick only a decay mode in particular if you want
        if selectDecay != -777 and selectDecay != genTauId:
            continue

        nGenTausType += 1
        foundGen = True

        # # # P4 Tau filters
        # if genVisTauP4.P() < 5:
        #     continue
        # if abs(math.cos(genVisTauP4.Theta()) > 0.9):
        #     continue

        # print ("Gen",genTauP4.P(),genVisTauP4.P(),genVisTauP4.Theta(),genVisTauP4.Phi(),genTauId,genTauQ,genTauDR,genTauNConsts)

        # Fill histograms
        hGenTauPt.Fill(genTauP4.Pt())  # Transverse momentum
        hGenVisTauPt.Fill(genVisTauP4.Pt())  # Visible transverse momentum
        hGenTauP.Fill(genTauP4.P())  # Momentum
        hGenVisTauP.Fill(genVisTauP4.P())  # Visible momentum
        hGenVisTauMass.Fill(genVisTauP4.M())  # Visible mass
        hGenTauType.Fill(genTauId)  # Tau decay type
        hGenTauQ.Fill(genTauQ)  # Tau charge
        hGenTauEta.Fill(genTauP4.Eta())  # Pseudo-rapidity
        hGenTauTheta.Fill(genTauP4.Theta())  # Theta angle
        
        if genTauId == 0:
            hGenTauTheta0.Fill(genTauP4.Theta())
            hGenTauP0.Fill(genTauP4.P())
            hGenTauVisP0.Fill(genVisTauP4.P())
        elif genTauId == 1:
            hGenTauTheta1.Fill(genTauP4.Theta())
            hGenTauP1.Fill(genTauP4.P())
            hGenTauVisP1.Fill(genVisTauP4.P())
        elif genTauId == 2:
            hGenTauTheta2.Fill(genTauP4.Theta())
            hGenTauP2.Fill(genTauP4.P())
            hGenTauVisP2.Fill(genVisTauP4.P())
        elif genTauId == 10:
            hGenTauTheta10.Fill(genTauP4.Theta())
            hGenTauP10.Fill(genTauP4.P())
            hGenTauVisP10.Fill(genVisTauP4.P())
        
            
        
        hGenTauDR.Fill(genTauDR)  # Angle of Tau Constituents
        countPionsRun = 0

        # print ("all GEN")
        # Look inside the generator level tau: check the constituents (decay products)

        # Compare with reconstructed taus using angle matching
        findMatch, nTausType = tauReco.MatchRecoGenTau(
            genTaus[i], recoTaus, nTausType, maxDRMatch=dRMatch, selectDecay=selectDecay
        )
        # For each generator level tau, find the reconstructed tau that is closest:
        if not matched_cm:
            true_predicted_label["GenID"].append(str(eventid) + str(i))
            true_predicted_label["True"].append(genTauId)

        # If you have not found it, continue: this is a efficiency loss
        if findMatch == -1:
            logger_process.debug("No match found for gen tau %s", genTaus[i])

            if nGenTaus > 2 and i <=1:
                result_labels[f"id-tau{i+1}"].append(-999)
            elif nGenTaus <= 2:
                result_labels[f"id-tau{i+1}"].append(-2)
            
            if not matched_cm:
                # true_predicted_label["Predicted"].append(-1)
                true_predicted_label["Predicted"].append(-2)
                true_predicted_label["PhotonPredicted"].append(-2)
            continue

        logger_process.debug("Found matched tau. Details:\n%s", recoTaus[findMatch])
        if matched_cm:
            true_predicted_label["GenID"].append(str(eventid) + str(i))
            true_predicted_label["True"].append(genTauId)
        # now, get the kinematics of the matched reco tau
        recoTauP4 = recoTaus[findMatch].getMomentum()
        recoTauId = recoTaus[findMatch].getID()
        recoTauQ = recoTaus[findMatch].getCharge()
        recoTauDR = recoTaus[findMatch].getMaxCone()
        recoTauConsts = recoTaus[findMatch].getDaughters()
        recoTauNConsts = recoTaus[findMatch].getnConst()

        # Reassign the recoTauId to the recoDM
        # This need to be checked as there exist other ID at Gen Level
        recoDM = recoTauId
        n_pi0s = 0
        
        if recoTauId < 10 and recoTauId >= 0:
            nPhotons = recoTauId
            n_pi0s = math.ceil(nPhotons / 2)
            recoDM = n_pi0s
        elif recoTauId >= 10:
            nPhotons = recoTauId - 10
            n_pi0s = math.ceil(nPhotons / 2)
            recoDM = 10 + n_pi0s
        
        if nGenTaus > 2 and i <=1:
            result_labels[f"id-tau{i+1}"].append(-999)
        elif nGenTaus <= 2:
            result_labels[f"id-tau{i+1}"].append(recoDM)

        
        true_predicted_label["Predicted"].append(recoDM)
        true_predicted_label["PhotonPredicted"].append(recoTauId)

        hMatchedTausPRes.Fill((recoTauP4.P() - genTauP4.P()) / genTauP4.P())
        hMatchedTausPtRes.Fill((recoTauP4.Pt() - genTauP4.Pt()) / genTauP4.Pt())
        # hMatchedTausChargeRes.Fill(abs(recoTauQ) - abs(genTauQ) / abs(genTauQ))
        hMatchedTausMaxAngleRes.Fill((recoTauDR - genTauDR) / genTauDR)
        hMatchedTausNCompRes.Fill((recoTauNConsts - genTauNConsts) / genTauNConsts)

        #          print ("Reco?",recoTauP4.P(),recoTauId,recoTauQ,recoTauDR,recoTauNConsts)

        # Now that we have a matched (gen,reco) pair, more checks for efficiency and resolution

        countPionsRun = 0
        # print ("Matched GEN!")
        # GEN: Look inside the tau, constituents:


        countPionsRun = 0
        # Init empyu TLorentzVector for the photon momentum
        photonCumulativeP4 = ROOT.TLorentzVector()
        photonCumulativeP4.SetXYZM(0, 0, 0, 0)
        # RECO:  Look inside the tau, constituents:
        # Filling histograms for the matched tau with reco level information
        recoTaus_photons = {}
        n_photons = 0
        for c in range(0, recoTauNConsts):
            const = recoTauConsts[c]
            constP4 = ROOT.TLorentzVector()
            try:
                constP4.SetXYZM(
                    const.getMomentum().x,
                    const.getMomentum().y,
                    const.getMomentum().z,
                    const.getMass(),
                )
            except AttributeError:
                # If the const does not have a mass, we assume it is a photon
                constP4.SetXYZM(
                    const.getMomentum().X(),
                    const.getMomentum().Y(),
                    const.getMomentum().Z(),
                    const.getMass(),
                )

            # This may be an error as we are confusing photons with neutrons

            logger_pi0mass.debug(f"Evaluating const to get photon with PDGID {const.getPDG()} and charge {const.getCharge()}")
            if const.getCharge() == 0:
                photonCumulativeP4 += constP4
                
                recoTaus_photons[n_photons] = const
                n_photons += 1
            
        
        
        
        if n_pi0s > 0:
            logger_pi0mass.debug(
                f"Found {n_pi0s} pi0s ({n_photons} photons) in the matched reco with Id {recoTauId} tau with real Id {genTauId}"
            )

            #  Rho decay Pi P
            if  genTauId == 1 and recoTauId == 2:
                photonCumP4 = ROOT.TLorentzVector()
                photonCumP4.SetXYZM(0, 0, 0, 0)
                
                for c in range(0, recoTauNConsts):
                    const = recoTauConsts[c]
                    constP4 = ROOT.TLorentzVector()
                    try:
                        constP4.SetXYZM(
                            const.getMomentum().x,
                            const.getMomentum().y,
                            const.getMomentum().z,
                            const.getMass(),
                        )
                    except AttributeError:
                        # If the const does not have a mass, we assume it is a photon
                        constP4.SetXYZM(
                            const.getMomentum().X(),
                            const.getMomentum().Y(),
                            const.getMomentum().Z(),
                            const.getMass(),
                        )
                    if const.getCharge() == 0:
                        hRecoRhoTwoPhotonDecayP.Fill(constP4.P())
                        photonCumP4 += constP4
                    else:
                        hRecoRhoPiDecayP.Fill(constP4.P())
                        pionP4 = constP4
                        
                hRecoRhoTwoPhotonDecaySumP.Fill(photonCumP4.P())
                h2DRecoRhoTwoPhotonDecayPiPhotonSumP.Fill(
                    pionP4.P(), photonCumP4.P()
                )
                # Gen level
                photonCumP4 = ROOT.TLorentzVector()
                photonCumP4.SetXYZM(0, 0, 0, 0)
                for c in range(0, genTauNConsts):
                    const = genTauConsts[c]
                    constP4 = ROOT.TLorentzVector()
                    try:
                        constP4.SetXYZM(
                            const.getMomentum().x,
                            const.getMomentum().y,
                            const.getMomentum().z,
                            const.getMass(),
                        )
                    except AttributeError:
                        # If the const does not have a mass, we assume it is a photon
                        constP4.SetXYZM(
                            const.getMomentum().X(),
                            const.getMomentum().Y(),
                            const.getMomentum().Z(),
                            const.getMass(),
                        )
                    if const.getCharge() == 0:
                        dau = const.getDaughters()
                        for dau in const.getDaughters():
                            photonP = ROOT.TLorentzVector()
                            try:
                                photonP.SetXYZM(
                                    dau.getMomentum().x,
                                    dau.getMomentum().y,
                                    dau.getMomentum().z,
                                    dau.getMass(),
                                )
                            except AttributeError:
                                # If the dau does not have a mass, we assume it is a photon
                                photonP.SetXYZM(
                                    dau.getMomentum().X(),
                                    dau.getMomentum().Y(),
                                    dau.getMomentum().Z(),
                                    dau.getMass(),
                                )
                            hGenRhoTwoPhotonDecayP.Fill(photonP.P())
                            photonCumP4 += photonP
                    else:
                        hGenRhoPiDecayP.Fill(constP4.P())
                        pionP4 = constP4
                hGenRhoTwoPhotonDecaySumP.Fill(photonCumP4.P())
                h2DGenRhoTwoPhotonDecayPiPhotonSumP.Fill(
                    pionP4.P(), photonCumP4.P()
                )
            if  recoTauId == 1 and genTauId == 1:
                for c in range(0, recoTauNConsts):
                    const = recoTauConsts[c]
                    constP4 = ROOT.TLorentzVector()
                    try:
                        constP4.SetXYZM(
                            const.getMomentum().x,
                            const.getMomentum().y,
                            const.getMomentum().z,
                            const.getMass(),
                        )
                    except AttributeError:
                        constP4.SetXYZM(
                            const.getMomentum().X(),
                            const.getMomentum().Y(),
                            const.getMomentum().Z(),
                            const.getMass(),
                        )
                    if const.getCharge() == 0:
                        photonP = constP4
                        hRecoRhoOnePhotonDecayPhotonP.Fill(constP4.P())
                    else:
                        hRecoRhoOnePhotonDecayPiP.Fill(constP4.P())
                        pionP4 = constP4
                h2DRecoRhoOnePhotonDecayPiPhotonP.Fill(pionP4.P(), photonP.P())
                # Gen level
                photonCumP4 = ROOT.TLorentzVector()
                photonCumP4.SetXYZM(0, 0, 0, 0)
                photonsP = []
                for c in range(0, genTauNConsts):
                    const = genTauConsts[c]
                    constP4 = ROOT.TLorentzVector()
                    try:
                        constP4.SetXYZM(
                            const.getMomentum().x,
                            const.getMomentum().y,
                            const.getMomentum().z,
                            const.getMass(),
                        )
                    except AttributeError:
                        constP4.SetXYZM(
                            const.getMomentum().X(),
                            const.getMomentum().Y(),
                            const.getMomentum().Z(),
                            const.getMass(),
                        )
                    if const.getPDG() == 111:
                        pi0daughters = const.getDaughters()
                        for dau in pi0daughters:
                            photonP = ROOT.TLorentzVector()
                            try:
                                photonP.SetXYZM(
                                    dau.getMomentum().x,
                                    dau.getMomentum().y,
                                    dau.getMomentum().z,
                                    dau.getMass(),
                                )
                            except AttributeError:
                                photonP.SetXYZM(
                                    dau.getMomentum().X(),
                                    dau.getMomentum().Y(),
                                    dau.getMomentum().Z(),
                                    dau.getMass(),
                                )
                            hGenRhoOnePhotonDecayPhotonP.Fill(photonP.P())
                            photonCumP4 += photonP
                            photonsP.append(photonP)
                    else:
                        hGenRhoOnePhotonDecayPiP.Fill(constP4.P())
                        pionP4 = constP4
                        
                hGenRhoOnePhotonDecayPhotonSumP.Fill(photonCumP4.P())
                h2DGenRhoOnePhotonDecayPiPhotonSumP.Fill(pionP4.P(), photonCumP4.P())
                ang = myutils.dRAngle(photonsP[0], photonsP[1])
                hGenRhoOnePhotonDecayPhotonAng.Fill(ang)

            
            if n_photons == 3 and (genTauId == 2 or genTauId == 1):
                logger_pi0mass.debug(
                    f"Found 3 photons in the matched reco tau with real Id {genTauId}"
                )
                pi0Mass_strmass, noMatchedPhotons = pi0Reco.getPi0Mass(recoTaus_photons, strategy = {"mass":-1})
                if pi0Mass_strmass:
                    hRecoConstPi0MassFromPhotonMasstr.Fill(pi0Mass_strmass)
                
                if noMatchedPhotons:
                    for ide, photon in noMatchedPhotons.items(): 
                        PhotonP4 = ROOT.TLorentzVector()
                        try:
                            PhotonP4.SetXYZM(
                                photon.getMomentum().x,
                                photon.getMomentum().y,
                                photon.getMomentum().z,
                                photon.getMass(),
                            )
                        except AttributeError:
                            # If the photon does not have a mass, we assume it is a photon
                            PhotonP4.SetXYZM(
                                photon.getMomentum().X(),
                                photon.getMomentum().Y(),
                                photon.getMomentum().Z(),
                                photon.getMass(),
                            )
                        if genTauId == 2:
                            hRecoConstlessPhotonPa1strMass.Fill(PhotonP4.P())
                            hRecoConstlessPhotonPa1strMassZoom.Fill(PhotonP4.P())
                            
                        elif genTauId == 1:
                            hRecoConstxtraPhotonPrhostrMass.Fill(PhotonP4.P())
                            hRecoConstxtraPhotonPrhostrMassZoom.Fill(PhotonP4.P())
                            
                    matched_keys = [key for key in range(3) if key not in noMatchedPhotons.keys()]
                    first_matched_P = ROOT.TLorentzVector()
                    try:
                        first_matched_P.SetXYZM(
                            recoTaus_photons[matched_keys[0]].getMomentum().x,
                            recoTaus_photons[matched_keys[0]].getMomentum().y,
                            recoTaus_photons[matched_keys[0]].getMomentum().z,
                            recoTaus_photons[matched_keys[0]].getMass(),
                        )
                        second_matched_P = ROOT.TLorentzVector()
                        second_matched_P.SetXYZM(
                            recoTaus_photons[matched_keys[1]].getMomentum().x,
                            recoTaus_photons[matched_keys[1]].getMomentum().y,
                            recoTaus_photons[matched_keys[1]].getMomentum().z,
                            recoTaus_photons[matched_keys[1]].getMass(),
                        )
                    except AttributeError:
                        # If the photon does not have a mass, we assume it is a photon
                        first_matched_P.SetXYZM(
                            recoTaus_photons[matched_keys[0]].getMomentum().X(),
                            recoTaus_photons[matched_keys[0]].getMomentum().Y(),
                            recoTaus_photons[matched_keys[0]].getMomentum().Z(),
                            recoTaus_photons[matched_keys[0]].getMass(),
                        )
                        second_matched_P = ROOT.TLorentzVector()
                        second_matched_P.SetXYZM(
                            recoTaus_photons[matched_keys[1]].getMomentum().X(),
                            recoTaus_photons[matched_keys[1]].getMomentum().Y(),
                            recoTaus_photons[matched_keys[1]].getMomentum().Z(),
                            recoTaus_photons[matched_keys[1]].getMass(),
                        )
                    non_matched_P = PhotonP4
                    # Moment of the photons
                    hRecoThreePhotonMatchOnestrMassP.Fill(first_matched_P.P())
                    hRecoThreePhotonMatchTwostrMassP.Fill(second_matched_P.P())
                    hRecoThreePhotonNoMatchstrMassP.Fill(non_matched_P.P())
                
                pi0Mass_strdist, noMatchedPhoton = pi0Reco.getPi0Mass(recoTaus_photons, strategy = {"distance":-1})
                if pi0Mass_strdist:
                    hRecoConstPi0MassFromPhotonDiststr.Fill(pi0Mass_strdist)
                if noMatchedPhoton:
                    for ide, photon in noMatchedPhotons.items(): 
                        PhotonP4 = ROOT.TLorentzVector()
                        try:
                            PhotonP4.SetXYZM(
                                photon.getMomentum().x,
                                photon.getMomentum().y,
                                photon.getMomentum().z,
                                photon.getMass(),
                            )
                        except AttributeError:
                            # If the photon does not have a mass, we assume it is a photon
                            PhotonP4.SetXYZM(
                                photon.getMomentum().X(),
                                photon.getMomentum().Y(),
                                photon.getMomentum().Z(),
                                photon.getMass(),
                            )
                        if genTauId == 2:
                            hRecoConstlessPhotonPa1strDist.Fill(PhotonP4.P())
                        elif genTauId == 1:
                            hRecoConstxtraPhotonPrhostrDist.Fill(PhotonP4.P())
                    matched_keys = [key for key in range(3) if key not in noMatchedPhotons.keys()]
                    first_matched_P = ROOT.TLorentzVector()
                    try:
                        first_matched_P.SetXYZM(
                            recoTaus_photons[matched_keys[0]].getMomentum().x,
                            recoTaus_photons[matched_keys[0]].getMomentum().y,
                            recoTaus_photons[matched_keys[0]].getMomentum().z,
                            recoTaus_photons[matched_keys[0]].getMass(),
                        )
                        second_matched_P = ROOT.TLorentzVector()
                        second_matched_P.SetXYZM(
                            recoTaus_photons[matched_keys[1]].getMomentum().x,
                            recoTaus_photons[matched_keys[1]].getMomentum().y,
                            recoTaus_photons[matched_keys[1]].getMomentum().z,
                            recoTaus_photons[matched_keys[1]].getMass(),
                        )
                    except AttributeError:
                        # If the photon does not have a mass, we assume it is a photon
                        first_matched_P.SetXYZM(
                            recoTaus_photons[matched_keys[0]].getMomentum().X(),
                            recoTaus_photons[matched_keys[0]].getMomentum().Y(),
                            recoTaus_photons[matched_keys[0]].getMomentum().Z(),
                            recoTaus_photons[matched_keys[0]].getMass(),
                        )
                        second_matched_P = ROOT.TLorentzVector()
                        second_matched_P.SetXYZM(
                            recoTaus_photons[matched_keys[1]].getMomentum().X(),
                            recoTaus_photons[matched_keys[1]].getMomentum().Y(),
                            recoTaus_photons[matched_keys[1]].getMomentum().Z(),
                            recoTaus_photons[matched_keys[1]].getMass(),
                        )
                    non_matched_P = PhotonP4
                    # Moment of the photons
                    hRecoThreePhotonMatchOnestrDistP.Fill(first_matched_P.P())
                    hRecoThreePhotonMatchTwostrDistP.Fill(second_matched_P.P())
                    hRecoThreePhotonNoMatchstrDistP.Fill(non_matched_P.P())
                            
            elif n_photons == 1 and (genTauId == 0 or genTauId == 1):
                logger_pi0mass.debug(
                    f"Found 1 photons in the matched reco tau with real Id {genTauId}"
                )
                if genTauId == 0:
                    hRecoConstxtraPhotonPi.Fill(photonCumulativeP4.P())
                    hRecoConstxtraPhotonPiZoom.Fill(photonCumulativeP4.P())
                elif genTauId == 1:
                    hRecoConstlessPhotonPrho.Fill(photonCumulativeP4.P())
                    hRecoConstlessPhotonPrhoZoom.Fill(photonCumulativeP4.P())
                    
            
            elif n_photons == 2 and (genTauId == 1):
                logger_pi0mass.debug(
                    f"Found 2 photons in the matched reco tau with real Id {genTauId}"
                )
                hRecoConstPi0Mass.Fill(photonCumulativeP4.M())
                photon1_P4 = ROOT.TLorentzVector()
                try:
                    photon1_P4.SetXYZM(
                        recoTaus_photons[0].getMomentum().x,
                        recoTaus_photons[0].getMomentum().y,
                        recoTaus_photons[0].getMomentum().z,
                        recoTaus_photons[0].getMass(),
                    )
                    photon2_P4 = ROOT.TLorentzVector()
                    photon2_P4.SetXYZM(
                        recoTaus_photons[1].getMomentum().x,
                        recoTaus_photons[1].getMomentum().y,
                        recoTaus_photons[1].getMomentum().z,
                        recoTaus_photons[1].getMass(),
                    )
                except AttributeError:
                    # If the photon does not have a mass, we assume it is a photon
                    photon1_P4.SetXYZM(
                        recoTaus_photons[0].getMomentum().X(),
                        recoTaus_photons[0].getMomentum().Y(),
                        recoTaus_photons[0].getMomentum().Z(),
                        recoTaus_photons[0].getMass(),
                    )
                    photon2_P4 = ROOT.TLorentzVector()
                    photon2_P4.SetXYZM(
                        recoTaus_photons[1].getMomentum().X(),
                        recoTaus_photons[1].getMomentum().Y(),
                        recoTaus_photons[1].getMomentum().Z(),
                        recoTaus_photons[1].getMass(),
                    )
                ang_dist = myutils.dRAngle(
                    photon1_P4, photon2_P4
                )
                hRecoConstTwoPhotonAngDist.Fill(ang_dist)
            


        hRecoTauType.Fill(recoTauId)
        hRecoTauPt.Fill(recoTauP4.Pt())
        hRecoTauP.Fill(recoTauP4.P())

        hRecoTauMass.Fill(recoTauP4.M())
        hRecoTauQ.Fill(recoTauQ)
        hRecoTauEta.Fill(recoTauP4.Eta())
        hRecoTauTheta.Fill(recoTauP4.Theta())

        hRecoTauDR.Fill(recoTauDR)

        hMatchedGenTauPt.Fill(genTauP4.Pt())
        hMatchedGenVisTauPt.Fill(genVisTauP4.Pt())
        hMatchedGenTauP.Fill(genTauP4.P())
        hMatchedGenVisTauP.Fill(genVisTauP4.P())

        hMatchedGenVisTauMass.Fill(genVisTauP4.M())
        hMatchedGenTauType.Fill(genTauId)
        hMatchedGenTauQ.Fill(genTauQ)
        hMatchedGenTauEta.Fill(genTauP4.Eta())
        hMatchedGenTauTheta.Fill(genTauP4.Theta())

        hMatchedGenTauDR.Fill(genTauDR)

        h2DTauPt.Fill(recoTauP4.Pt(), genVisTauP4.Pt())
        h2DTauMass.Fill(recoTauP4.M(), genVisTauP4.M())
        h2DTauType.Fill(recoTauId, genTauId)
        h2DTauP.Fill(recoTauP4.P(), genVisTauP4.P())

        h2DTauDR.Fill(recoTauDR, genTauDR)

        if genTauId == 0:
            hMatchedGenTauP0.Fill(genTauP4.P())
            hMatchedGenTauVisP0.Fill(genVisTauP4.P())
            hMatchedGenTauTheta0.Fill(genTauP4.Theta())
            hTauThetaRes0.Fill(
                (recoTauP4.Theta() - genVisTauP4.Theta())
                / (genVisTauP4.Theta()+1e-10)
            )
            hTauPRes0.Fill(
                    (recoTauP4.P() - genVisTauP4.P()) / (genVisTauP4.P()+1e-10)
                )
            if recoDM == 0:
                hMatchedTauPRes0.Fill(
                    (recoTauP4.P() - genVisTauP4.P()) / (genVisTauP4.P()+1e-10)
                )
                hMatchedTauThetaRes0.Fill(
                    (recoTauP4.Theta() - genVisTauP4.Theta())
                    / (genVisTauP4.Theta()+1e-10)
                )
        elif genTauId == 1:
            hMatchedGenTauP1.Fill(genTauP4.P())
            hMatchedGenTauVisP1.Fill(genVisTauP4.P())
            hMatchedGenTauTheta1.Fill(genTauP4.Theta())
            hTauThetaRes1.Fill(
                (recoTauP4.Theta() - genVisTauP4.Theta())
                / (genVisTauP4.Theta()+1e-10)
            )
            hTauPRes1.Fill(
                    (recoTauP4.P() - genVisTauP4.P()) / (genVisTauP4.P()+1e-10)
                )
            if recoDM == 1:
                hMatchedTauPRes1.Fill(
                    (recoTauP4.P() - genVisTauP4.P()) / (genVisTauP4.P()+1e-10)
                )
                hMatchedTauThetaRes1.Fill(
                    (recoTauP4.Theta() - genVisTauP4.Theta())
                    / (genVisTauP4.Theta()+1e-10)
                )
        elif genTauId == 2:
            hMatchedGenTauP2.Fill(genTauP4.P())
            hMatchedGenTauVisP2.Fill(genVisTauP4.P())
            hMatchedGenTauTheta2.Fill(genTauP4.Theta())
            hTauThetaRes2.Fill(
                (recoTauP4.Theta() - genVisTauP4.Theta())
                / (genVisTauP4.Theta()+1e-10)
            )
            hTauPRes2.Fill(
                    (recoTauP4.P() - genVisTauP4.P()) / (genVisTauP4.P()+1e-10)
                )
            if recoDM == 2:
                hMatchedTauPRes2.Fill(
                    (recoTauP4.P() - genVisTauP4.P()) / (genVisTauP4.P()+1e-10)
                )
                hMatchedTauThetaRes2.Fill(
                    (recoTauP4.Theta() - genVisTauP4.Theta())
                    / (genVisTauP4.Theta()+1e-10)
                )
        elif genTauId == 10:
            hMatchedGenTauP10.Fill(genTauP4.P())
            hMatchedGenTauVisP10.Fill(genVisTauP4.P())
            hMatchedGenTauTheta10.Fill(genTauP4.Theta())
            hTauThetaRes10.Fill(
                (recoTauP4.Theta() - genVisTauP4.Theta())
                / (genVisTauP4.Theta()+1e-10)
            )
            hTauPRes10.Fill(
                    (recoTauP4.P() - genVisTauP4.P()) / (genVisTauP4.P()+1e-10)
                )
            if recoDM == 10:
                hMatchedTauPRes10.Fill(
                    (recoTauP4.P() - genVisTauP4.P()) / (genVisTauP4.P()+1e-10)
                )
                hMatchedTauThetaRes10.Fill(
                    (recoTauP4.Theta() - genVisTauP4.Theta())
                    / (genVisTauP4.Theta()+1e-10)
                )
        
        # Fill the histograms for the matched reco tau with gen level information
        if recoDM == 0:
            hRecoTauTheta0.Fill(recoTauP4.Theta())
            hRecoTauP0.Fill(recoTauP4.P())
        elif recoDM == 1:
            hRecoTauTheta1.Fill(recoTauP4.Theta())
            hRecoTauP1.Fill(recoTauP4.P())
        elif recoDM == 2:
            hRecoTauTheta2.Fill(recoTauP4.Theta())
            hRecoTauP2.Fill(recoTauP4.P())
        elif recoDM == 10:
            hRecoTauTheta10.Fill(recoTauP4.Theta())
            hRecoTauP10.Fill(recoTauP4.P())
        
        
        # Resolution plots:
        if genVisTauP4.P() != 0:
            hResTauPt.Fill((recoTauP4.Pt() - genVisTauP4.Pt()) / genVisTauP4.Pt())
            hResTauMass.Fill((recoTauP4.M() - genVisTauP4.M()) / genVisTauP4.M())
            hResTauP.Fill((recoTauP4.P() - genVisTauP4.P()) / genVisTauP4.P())

    # print ("Taus???",nGenTaus,nTaus)
    hNTaus.Fill(nRecoTaus)
    hNTausType.Fill(nTausType)
    hNGenTausType.Fill(nGenTausType)
    hNGenTaus.Fill(nGenTausHad)


# Do efficiencies (divide matched gen by all gen)
hEffiGenPi0Mass = hMatchedGenConstPi0Mass.Clone()
hEffiGenPi0Mass.SetName("hEffiGenPi0Mass")
hEffiGenPi0Mass.Divide(hGenConstPi0Mass)

hEffiGenTauPt = hMatchedGenTauPt.Clone()
hEffiGenTauPt.SetName("hEffiGenTauPt")
hEffiGenTauPt.Divide(hGenTauPt)

hEffiGenVisTauPt = hMatchedGenVisTauPt.Clone()
hEffiGenVisTauPt.SetName("hEffiGenVisTauPt")
hEffiGenVisTauPt.Divide(hGenVisTauPt)

hEffiGenTauP = hMatchedGenTauP.Clone()
hEffiGenTauP.SetName("hEffiGenTauP")
hEffiGenTauP.Divide(hGenTauP)

hEffiGenVisTauP = hMatchedGenVisTauP.Clone()
hEffiGenVisTauP.SetName("hEffiGenVisTauP")
hEffiGenVisTauP.Divide(hGenVisTauP)

hEffiGenVisTauMass = hMatchedGenVisTauMass.Clone()
hEffiGenVisTauMass.SetName("hEffiGenVisTauMass")
hEffiGenVisTauMass.Divide(hGenVisTauMass)

hEffiGenTauEta = hMatchedGenTauEta.Clone()
hEffiGenTauEta.SetName("hEffiGenTauEta")
hEffiGenTauEta.Divide(hGenTauEta)

hEffiGenTauTheta = hMatchedGenTauTheta.Clone()
hEffiGenTauTheta.SetName("hEffiGenTauTheta")
hEffiGenTauTheta.Divide(hGenTauTheta)

hEffiGenTauType = hMatchedGenTauType.Clone()
hEffiGenTauType.SetName("hEffiGenTauType")
hEffiGenTauType.Divide(hGenTauType)


# Theta angle per decay
hEffiGenTauTheta0 = hMatchedGenTauTheta0.Clone()
hEffiGenTauTheta0.SetName("hEffiGenTauTheta0")
hEffiGenTauTheta0.Divide(hGenTauTheta0)
hEffiGenTauTheta1 = hMatchedGenTauTheta1.Clone()
hEffiGenTauTheta1.SetName("hEffiGenTauTheta1")
hEffiGenTauTheta1.Divide(hGenTauTheta1)
hEffiGenTauTheta2 = hMatchedGenTauTheta2.Clone()
hEffiGenTauTheta2.SetName("hEffiGenTauTheta2")
hEffiGenTauTheta2.Divide(hGenTauTheta2)
hEffiGenTauTheta10 = hMatchedGenTauTheta10.Clone()
hEffiGenTauTheta10.SetName("hEffiGenTauTheta10")
hEffiGenTauTheta10.Divide(hGenTauTheta10)


# Momentum per decay
hEffiGenTauP0 = hMatchedGenTauP0.Clone()
hEffiGenTauP0.SetName("hEffiGenTauP0")
hEffiGenTauP0.Divide(hGenTauP0)
hEffiGenTauP1 = hMatchedGenTauP1.Clone()
hEffiGenTauP1.SetName("hEffiGenTauP1")
hEffiGenTauP1.Divide(hGenTauP1)
hEffiGenTauP2 = hMatchedGenTauP2.Clone()
hEffiGenTauP2.SetName("hEffiGenTauP2")
hEffiGenTauP2.Divide(hGenTauP2)
hEffiGenTauP10 = hMatchedGenTauP10.Clone()
hEffiGenTauP10.SetName("hEffiGenTauP10")
hEffiGenTauP10.Divide(hGenTauP10)

# Vis Momentum per decay
hEffiGenTauVisP0 = hMatchedGenTauVisP0.Clone()
hEffiGenTauVisP0.SetName("hEffiGenTauVisP0")
hEffiGenTauVisP0.Divide(hGenTauVisP0)
hEffiGenTauVisP1 = hMatchedGenTauVisP1.Clone()
hEffiGenTauVisP1.SetName("hEffiGenTauVisP1")
hEffiGenTauVisP1.Divide(hGenTauVisP1)
hEffiGenTauVisP2 = hMatchedGenTauVisP2.Clone()
hEffiGenTauVisP2.SetName("hEffiGenTauVisP2")
hEffiGenTauVisP2.Divide(hGenTauVisP2)
hEffiGenTauVisP10 = hMatchedGenTauVisP10.Clone()
hEffiGenTauVisP10.SetName("hEffiGenTauVisP10")
hEffiGenTauVisP10.Divide(hGenTauVisP10)






logger_process.info("Found %d events", countEvents)

decaystr = "decayAll" if selectDecay == -777 else "decay{}".format(selectDecay)
true_predicted_label_output_file = outputpath + f"true_predicted_label_{decaystr}.csv"
true_predicted_label_df = pd.DataFrame(true_predicted_label)
true_predicted_label_df.to_csv(true_predicted_label_output_file, index=False)

result_labels = pd.DataFrame(result_labels)
result_labels.to_csv(outputpath + "result_labels_pfo.csv", index=False)

# Check if config["output"]["outputlabels"] is a list
if type(config["output"]["outputlabels"]) is not list:
    if config["output"]["outputlabels"] is None:
        config["output"]["outputlabels"] = []
    else:
        config["output"]["outputlabels"] = [config["output"]["outputlabels"]]
if true_predicted_label_output_file not in config["output"]["outputlabels"]:
    config["output"]["outputlabels"].append(true_predicted_label_output_file)

output_config_file = outputpath + "config.yaml"
with open(output_config_file, "w") as file:
    yaml.dump(config, file)
    logger_io.info("Configuration file saved to %s", output_config_file)

outfile = ROOT.TFile(outputpath + fileOutName, "RECREATE")


# Theta
hGenTauTheta0.Write()
hGenTauTheta1.Write()
hGenTauTheta2.Write()
hGenTauTheta10.Write()
hMatchedGenTauTheta0.Write()
hMatchedGenTauTheta1.Write()
hMatchedGenTauTheta2.Write()
hMatchedGenTauTheta10.Write()
hRecoTauTheta0.Write()
hRecoTauTheta1.Write()
hRecoTauTheta2.Write()
hRecoTauTheta10.Write()

hTauThetaRes0.Write()
hTauThetaRes1.Write()
hTauThetaRes2.Write()
hTauThetaRes10.Write()

hMatchedTauThetaRes0.Write()
hMatchedTauThetaRes1.Write()
hMatchedTauThetaRes2.Write()
hMatchedTauThetaRes10.Write()

# P (Momentum)
hGenTauP0.Write() 
hGenTauP1.Write() 
hGenTauP2.Write() 
hGenTauP10.Write() 
hGenTauVisP0.Write() 
hGenTauVisP1.Write() 
hGenTauVisP2.Write() 
hGenTauVisP10.Write() 
hMatchedGenTauP0.Write() 
hMatchedGenTauP1.Write() 
hMatchedGenTauP2.Write() 
hMatchedGenTauP10.Write() 
hMatchedGenTauVisP0.Write() 
hMatchedGenTauVisP1.Write() 
hMatchedGenTauVisP2.Write() 
hMatchedGenTauVisP10.Write()

hRecoTauP0.Write()
hRecoTauP1.Write()
hRecoTauP2.Write()
hRecoTauP10.Write()

# Resolution P
hTauPRes0.Write()
hTauPRes1.Write() 
hTauPRes2.Write()
hTauPRes10.Write()

hMatchedTauPRes0.Write()
hMatchedTauPRes1.Write()
hMatchedTauPRes2.Write()
hMatchedTauPRes10.Write()

hEffiGenTauTheta0.Write()
hEffiGenTauTheta1.Write()
hEffiGenTauTheta2.Write()
hEffiGenTauTheta10.Write()
hEffiGenTauP0.Write()
hEffiGenTauP1.Write()
hEffiGenTauP2.Write()
hEffiGenTauP10.Write()
hEffiGenTauVisP0.Write()
hEffiGenTauVisP1.Write()
hEffiGenTauVisP2.Write()
hEffiGenTauVisP10.Write()




# Reco Mass and P from different strategies
hRecoConstPi0MassFromPhotonMasstr.Write()
hRecoConstPi0MassFromPhotonDiststr.Write()
hRecoConstlessPhotonPa1strMass.Write()
hRecoConstxtraPhotonPrhostrMass.Write()
hRecoConstlessPhotonPa1strDist.Write()
hRecoConstxtraPhotonPrhostrDist.Write()
hRecoConstlessPhotonPrho.Write()
hRecoConstxtraPhotonPi.Write()
hRecoConstPi0Mass.Write()
# Hist with zooms in 0,2
hRecoConstlessPhotonPa1strMassZoom.Write()
hRecoConstxtraPhotonPrhostrMassZoom.Write()
hRecoConstlessPhotonPrhoZoom.Write()
hRecoConstxtraPhotonPiZoom.Write()



# Matched photons moment (Distance strategy)
hRecoThreePhotonMatchOnestrDistP.Write()
hRecoThreePhotonMatchTwostrDistP.Write()
hRecoThreePhotonNoMatchstrDistP.Write()

# Matched photons moment (Mass strategy)
hRecoThreePhotonMatchOnestrMassP.Write()
hRecoThreePhotonMatchTwostrMassP.Write()
hRecoThreePhotonNoMatchstrMassP.Write()

hRecoConstTwoPhotonAngDist.Write()


#  Rho decay Pi P
hRecoRhoPiDecayP.Write() 
hGenRhoPiDecayP.Write()
# Rho decay two photons case P
hRecoRhoTwoPhotonDecayP.Write()
hGenRhoTwoPhotonDecayP.Write()
hRecoRhoTwoPhotonDecaySumP.Write()
hGenRhoTwoPhotonDecaySumP.Write()
# Hist of pi P vs sum of photons P
h2DRecoRhoTwoPhotonDecayPiPhotonSumP.Write()
h2DGenRhoTwoPhotonDecayPiPhotonSumP.Write()

# Rho False decay (one photon)
hRecoRhoOnePhotonDecayPiP.Write()
hGenRhoOnePhotonDecayPiP.Write()
hRecoRhoOnePhotonDecayPhotonP.Write()
hGenRhoOnePhotonDecayPhotonP.Write()
hGenRhoOnePhotonDecayPhotonSumP.Write()
# Angle between the photons at gen level
hGenRhoOnePhotonDecayPhotonAng.Write()
# Hist of pi P vs photon P at reco level:
h2DRecoRhoOnePhotonDecayPiPhotonP.Write()
# Hist of pi P vs photon P at gen level:
h2DGenRhoOnePhotonDecayPiPhotonSumP.Write()


hGenConstPi0Mass.Write()
hEffiGenPi0Mass.Write()
hMatchedGenConstPi0Mass.Write()
h2DPi0MassOverNPhoton.Write()


hGenTauPt.Write()
hGenVisTauPt.Write()
hGenTauP.Write()
hGenVisTauP.Write()
hGenTauType.Write()
hGenVisTauMass.Write()
hGenTauQ.Write()
hGenTauEta.Write()
hGenTauTheta.Write()

hGenTauDR.Write()

hMatchedGenTauPt.Write()
hMatchedGenVisTauPt.Write()
hMatchedGenTauP.Write()
hMatchedGenVisTauP.Write()

hMatchedGenTauType.Write()
hMatchedGenVisTauMass.Write()
hMatchedGenTauQ.Write()
hMatchedGenTauEta.Write()
hMatchedGenTauTheta.Write()

hMatchedGenTauDR.Write()


hEffiGenTauPt.Write()
hEffiGenVisTauPt.Write()
hEffiGenTauP.Write()
hEffiGenVisTauP.Write()

hEffiGenTauType.Write()
hEffiGenVisTauMass.Write()
hEffiGenTauEta.Write()
hEffiGenTauTheta.Write()


hRecoTauPt.Write()
hRecoTauP.Write()

hRecoTauType.Write()
hRecoTauMass.Write()
hRecoTauQ.Write()
hRecoTauEta.Write()
hRecoTauTheta.Write()

hRecoTauDR.Write()

h2DTauPt.Write()
h2DTauP.Write()

h2DTauMass.Write()
h2DTauType.Write()

h2DTauDR.Write()

hResTauPt.Write()
hResTauP.Write()

hResTauMass.Write()

hNTaus.Write()
hNGenTaus.Write()

hNTausType.Write()
hNGenTausType.Write()

hMatchedTausPRes.Write()
hMatchedTausPtRes.Write()
hMatchedTausChargeRes.Write()
hMatchedTausMaxAngleRes.Write()
hMatchedTausNCompRes.Write()

logger_io.info("Output file %s", outputpath + fileOutName)
logger_io.info("End of job")
outfile.Close()
