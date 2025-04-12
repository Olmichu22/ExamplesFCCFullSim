import sys, os, math
from array import array
import ROOT
from ROOT import TFile, TTree, TH1F, TH2F, std
import numpy as np
from podio import root_io
import edm4hep
from pathlib import Path
import pprint
import yaml
import pandas as pd

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
    default="True",
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
else:
    log_level = logging.DEBUG  # Debug messages for -vv or higher
    handlers=[logging.FileHandler(outputpath + "/" + "app.log", mode="w")]
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
path = "/pnfs/ciemat.es/data/cms/store/user/cepeda/FCC/FullSim/"
file = "out_reco_edm4hep_edm4hep"
filenames = []
dir_path = path + "/" + sample

nfiles = len(os.listdir(dir_path))

nfiles = 1000
if test == True:
    nfiles = 1

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

# Configs and reading finished
# ----------------------------------------------------------------------


# collections to use
genparts = "MCParticles"
pfobjects = "PandoraPFOs"
# pfobjects ="TightSelectedPandoraPFOs"

gen_tree = TTree("gen_tree", f"Tree {cut_string}_Sample_{sample}")
matched_tree = TTree("matched_tree", f"Reco Tree {cut_string}_Sample_{sample}")
gen_const_tree = TTree("gen_const_tree", f"Const Tree {cut_string}_Sample_{sample}")
reco_const_tree = TTree("reco_const_tree", f"Reco Const Tree {cut_string}_Sample_{sample}")
matched_gen_const_tree = TTree("matched_gen_const_tree", f"Matched Const Tree {cut_string}_Sample_{sample}")

photon_tree = TTree("photon_tree", f"Photon Tree {cut_string}_Sample_{sample}")

# Defining many histogram
hGenTauPt = np.array([0])
hGenVisTauPt = np.array([0])
hGenTauP = np.array([0])
hGenVisTauP = np.array([0])
hGenTauType = np.array([0], dtype =int)
hGenVisTauMass = np.array([0])
hGenTauQ = np.array([0])
hGenTauEta = np.array([0])
hGenTauTheta = np.array([0])
hGenTauDR = np.array([0])

gen_tree.Branch("hGenTauPt", hGenTauPt, "hGenTauPt/F")
gen_tree.Branch("hGenVisTauPt", hGenVisTauPt, "hGenVisTauPt/F")
gen_tree.Branch("hGenTauP", hGenTauP, "hGenTauP/F")
gen_tree.Branch("hGenVisTauP", hGenVisTauP, "hGenVisTauP/F")
gen_tree.Branch("hGenTauType", hGenTauType, "hGenTauType/I")
gen_tree.Branch("hGenVisTauMass", hGenVisTauMass, "hGenVisTauMass/F")
gen_tree.Branch("hGenTauQ", hGenTauQ, "hGenTauQ/F")
gen_tree.Branch("hGenTauEta", hGenTauEta, "hGenTauEta/F")
gen_tree.Branch("hGenTauTheta", hGenTauTheta, "hGenTauTheta/F")
gen_tree.Branch("hGenTauDR", hGenTauDR, "hGenTauDR/F")


hMatchedGenTauPt = np.array([0])
hMatchedGenVisTauPt = np.array([0])
hMatchedGenTauP = np.array([0])
hMatchedGenVisTauP = np.array([0])
matched_tree.Branch("hMatchedGenTauPt", hMatchedGenTauPt, "hMatchedGenTauPt/F")
matched_tree.Branch("hMatchedGenVisTauPt", hMatchedGenVisTauPt, "hMatchedGenVisTauPt/F")
matched_tree.Branch("hMatchedGenTauP", hMatchedGenTauP, "hMatchedGenTauP/F")
matched_tree.Branch("hMatchedGenVisTauP", hMatchedGenVisTauP, "hMatchedGenVisTauP/F")


hMatchedGenTauType = np.array([0], dtype =int)
hMatchedGenVisTauMass = np.array([0])
hMatchedGenTauQ = np.array([0])
hMatchedGenTauEta = np.array([0])
hMatchedGenTauTheta = np.array([0])
hMatchedGenTauDR = np.array([0])
matched_tree.Branch("hMatchedGenTauType", hMatchedGenTauType, "hMatchedGenTauType/I")
matched_tree.Branch("hMatchedGenVisTauMass", hMatchedGenVisTauMass, "hMatchedGenVisTauMass/F")
matched_tree.Branch("hMatchedGenTauQ", hMatchedGenTauQ, "hMatchedGenTauQ/F")
matched_tree.Branch("hMatchedGenTauEta", hMatchedGenTauEta, "hMatchedGenTauEta/F")
matched_tree.Branch("hMatchedGenTauTheta", hMatchedGenTauTheta, "hMatchedGenTauTheta/F")
matched_tree.Branch("hMatchedGenTauDR", hMatchedGenTauDR, "hMatchedGenTauDR/F")


hRecoTauPt = np.array([0])
hRecoTauP = np.array([0])
matched_tree.Branch("hRecoTauPt", hRecoTauPt, "hRecoTauPt/F")
matched_tree.Branch("hRecoTauP", hRecoTauP, "hRecoTauP/F")

hRecoTauMass = np.array([0])
hRecoTauType = np.array([0], dtype =int)
hRecoTauQ = np.array([0])
hRecoTauEta = np.array([0])
hRecoTauTheta = np.array([0])
hRecoTauDR = np.array([0])
matched_tree.Branch("hRecoTauMass", hRecoTauMass, "hRecoTauMass/F")
matched_tree.Branch("hRecoTauType", hRecoTauType, "hRecoTauType/I")
matched_tree.Branch("hRecoTauQ", hRecoTauQ, "hRecoTauQ/F")
matched_tree.Branch("hRecoTauEta", hRecoTauEta, "hRecoTauEta/F")
matched_tree.Branch("hRecoTauTheta", hRecoTauTheta, "hRecoTauTheta/F")
matched_tree.Branch("hRecoTauDR", hRecoTauDR, "hRecoTauDR/F")


hRecoConstP = np.array([0])
hRecoConstPhotonP = np.array([0])
hRecoConstPionP = np.array([0])
reco_const_tree.Branch("hRecoConstP", hRecoConstP, "hRecoConstP/F")
reco_const_tree.Branch("hRecoConstPhotonP", hRecoConstPhotonP, "hRecoConstPhotonP/F")
reco_const_tree.Branch("hRecoConstPionP", hRecoConstPionP, "hRecoConstPionP/F")



hRecoConstPOverTauP = np.array([0]) 
hRecoConstPhotonPOverTauP = np.array([0])
hRecoConstPionPOverTauP = np.array([0])
reco_const_tree.Branch("hRecoConstPOverTauP", hRecoConstPOverTauP, "hRecoConstPOverTauP/F")
reco_const_tree.Branch("hRecoConstPhotonPOverTauP", hRecoConstPhotonPOverTauP, "hRecoConstPhotonPOverTauP/F")
reco_const_tree.Branch("hRecoConstPionPOverTauP", hRecoConstPionPOverTauP, "hRecoConstPionPOverTauP/F")


hRecoConstPion1P = np.array([0])
hRecoConstPion2P = np.array([0])
hRecoConstPion3P = np.array([0])
reco_const_tree.Branch("hRecoConstPion1P", hRecoConstPion1P, "hRecoConstPion1P/F")
reco_const_tree.Branch("hRecoConstPion2P", hRecoConstPion2P, "hRecoConstPion2P/F")
reco_const_tree.Branch("hRecoConstPion3P", hRecoConstPion3P, "hRecoConstPion3P/F")


hRecoConstPi0Mass = np.array([0])
hRecoConstTwoPhotonAngDist = np.array([0])
photon_tree.Branch("hRecoConstPi0Mass", hRecoConstPi0Mass, "hRecoConstPi0Mass/F")
photon_tree.Branch("hRecoConstTwoPhotonAngDist", hRecoConstTwoPhotonAngDist, "hRecoConstTwoPhotonAngDist/F")

hRecoConstPi0MassFromPhotonMasstr = np.array([0])
hRecoConstPi0MassFromPhotonDiststr = np.array([0])
photon_tree.Branch("hRecoConstPi0MassFromPhotonMasstr", hRecoConstPi0MassFromPhotonMasstr, "hRecoConstPi0MassFromPhotonMasstr/F")
photon_tree.Branch("hRecoConstPi0MassFromPhotonDiststr", hRecoConstPi0MassFromPhotonDiststr, "hRecoConstPi0MassFromPhotonDiststr/F")

hGenConstPi0Mass = np.array([0])
hMatchedGenConstPi0Mass = np.array([0])
gen_const_tree.Branch("hGenConstPi0Mass", hGenConstPi0Mass, "hGenConstPi0Mass/F")
matched_tree.Branch("hMatchedGenConstPi0Mass", hMatchedGenConstPi0Mass, "hMatchedGenConstPi0Mass/F")

hMatchedGenConstPion1P = np.array([0])
hMatchedGenConstPion2P = np.array([0])
hMatchedGenConstPion3P = np.array([0])
matched_gen_const_tree.Branch("hMatchedGenConstPion1P", hMatchedGenConstPion1P, "hMatchedGenConstPion1P/F")
matched_gen_const_tree.Branch("hMatchedGenConstPion2P", hMatchedGenConstPion2P, "hMatchedGenConstPion2P/F")
matched_gen_const_tree.Branch("hMatchedGenConstPion3P", hMatchedGenConstPion3P, "hMatchedGenConstPion3P/F")


hGenConstPion1P = np.array([0])
hGenConstPion2P = np.array([0])
hGenConstPion3P = np.array([0])
gen_const_tree.Branch("hGenConstPion1P", hGenConstPion1P, "hGenConstPion1P/F")
gen_const_tree.Branch("hGenConstPion2P", hGenConstPion2P, "hGenConstPion2P/F")
gen_const_tree.Branch("hGenConstPion3P", hGenConstPion3P, "hGenConstPion3P/F")

hGenConstP = np.array([0])
hGenConstPi0P = np.array([0])
hGenConstPionP = np.array([0])
# hGenConstPhotonP = np.array([0])
gen_const_tree.Branch("hGenConstP", hGenConstP, "hGenConstP/F")
gen_const_tree.Branch("hGenConstPi0P", hGenConstPi0P, "hGenConstPi0P/F")
gen_const_tree.Branch("hGenConstPionP", hGenConstPionP, "hGenConstPionP/F")
# gen_const_tree.Branch("hGenConstPhotonP", hGenConstPhotonP, "hGenConstPhotonP/F")

# Hist for reconstructed photons in a1 (pi pi0 pi0), rho (pi0 pi0), and pi cases
# 3 photon cases
hRecoConstlessPhotonPa1strMass = np.array([0])
hRecoConstlessPhotonPa1strDist = np.array([0])
photon_tree.Branch("hRecoConstlessPhotonPa1strMass", hRecoConstlessPhotonPa1strMass, "hRecoConstlessPhotonPa1strMass/F")
photon_tree.Branch("hRecoConstlessPhotonPa1strDist", hRecoConstlessPhotonPa1strDist, "hRecoConstlessPhotonPa1strDist/F")

hRecoConstxtraPhotonPrhostrMass = np.array([0])
hRecoConstxtraPhotonPrhostrDist = np.array([0])
photon_tree.Branch("hRecoConstxtraPhotonPrhostrMass", hRecoConstxtraPhotonPrhostrMass, "hRecoConstxtraPhotonPrhostrMass/F")
photon_tree.Branch("hRecoConstxtraPhotonPrhostrDist", hRecoConstxtraPhotonPrhostrDist, "hRecoConstxtraPhotonPrhostrDist/F")

# 1 photon cases
hRecoConstlessPhotonPrho = np.array([0])
hRecoConstxtraPhotonPi = np.array([0])
photon_tree.Branch("hRecoConstlessPhotonPrho", hRecoConstlessPhotonPrho, "hRecoConstlessPhotonPrho/F")
photon_tree.Branch("hRecoConstxtraPhotonPi", hRecoConstxtraPhotonPi, "hRecoConstxtraPhotonPi/F")

hRecoThreePhotonMatchOnestrMassP = np.array([0])
hRecoThreePhotonMatchOnestrDistP = np.array([0])
hRecoThreePhotonMatchTwostrMassP = np.array([0])
hRecoThreePhotonMatchTwostrDistP = np.array([0])
hRecoThreePhotonNoMatchstrMassP = np.array([0])
hRecoThreePhotonNoMatchstrDistP = np.array([0])
photon_tree.Branch("hRecoThreePhotonMatchOnestrMassP", hRecoThreePhotonMatchOnestrMassP, "hRecoThreePhotonMatchOnestrMassP/F")
photon_tree.Branch("hRecoThreePhotonMatchOnestrDistP", hRecoThreePhotonMatchOnestrDistP, "hRecoThreePhotonMatchOnestrDistP/F")
photon_tree.Branch("hRecoThreePhotonMatchTwostrMassP", hRecoThreePhotonMatchTwostrMassP, "hRecoThreePhotonMatchTwostrMassP/F")
photon_tree.Branch("hRecoThreePhotonMatchTwostrDistP", hRecoThreePhotonMatchTwostrDistP, "hRecoThreePhotonMatchTwostrDistP/F")
photon_tree.Branch("hRecoThreePhotonNoMatchstrMassP", hRecoThreePhotonNoMatchstrMassP, "hRecoThreePhotonNoMatchstrMassP/F")
photon_tree.Branch("hRecoThreePhotonNoMatchstrDistP", hRecoThreePhotonNoMatchstrDistP, "hRecoThreePhotonNoMatchstrDistP/F")




hGenConstPOverTauP = np.array([0])
hGenConstPi0POverTauP = np.array([0])
hGenConstPionPOverTauP = np.array([0])
# hGenConstPhotonPOverTauP = np.array([0])
gen_const_tree.Branch("hGenConstPOverTauP", hGenConstPOverTauP, "hGenConstPOverTauP/F")
gen_const_tree.Branch("hGenConstPi0POverTauP", hGenConstPi0POverTauP, "hGenConstPi0POverTauP/F")
gen_const_tree.Branch("hGenConstPionPOverTauP", hGenConstPionPOverTauP, "hGenConstPionPOverTauP/F")
# gen_const_tree.Branch("hGenConstPhotonPOverTauP", hGenConstPhotonPOverTauP, "hGenConstPhotonPOverTauP/F")


hMatchedGenConstP = np.array([0])
hMatchedGenConstPi0P = np.array([0])
hMatchedGenConstPionP = np.array([0])
hMatchedGenConstPOverTauP = np.array([0])
hMatchedGenConstPi0POverTauP = np.array([0])
hMatchedGenConstPionPOverTauP = np.array([0])
matched_gen_const_tree.Branch("hMatchedGenConstP", hMatchedGenConstP, "hMatchedGenConstP/F")
matched_gen_const_tree.Branch("hMatchedGenConstPi0P", hMatchedGenConstPi0P, "hMatchedGenConstPi0P/F")
matched_gen_const_tree.Branch("hMatchedGenConstPionP", hMatchedGenConstPionP, "hMatchedGenConstPionP/F")
matched_gen_const_tree.Branch("hMatchedGenConstPOverTauP", hMatchedGenConstPOverTauP, "hMatchedGenConstPOverTauP/F")
matched_gen_const_tree.Branch("hMatchedGenConstPi0POverTauP", hMatchedGenConstPi0POverTauP, "hMatchedGenConstPi0POverTauP/F")
matched_tree.Branch("hMatchedGenConstPionPOverTauP", hMatchedGenConstPionPOverTauP, "hMatchedGenConstPionPOverTauP/F")

hGenTauNConsts = np.array([0], dtype =int)
gen_tree.Branch("hGenTauNConsts", hGenTauNConsts, "hGenTauNConsts/I")

hResTauPt = np.array([0])
hResTauP = np.array([0])
hResTauMass = np.array([0])
matched_tree.Branch("hResTauPt", hResTauPt, "hResTauPt/F")
matched_tree.Branch("hResTauP", hResTauP, "hResTauP/F")
matched_tree.Branch("hResTauMass", hResTauMass, "hResTauMass/F")


# hNTaus = np.array([0])
# hNGenTaus = np.array([0])
# hNTausType = np.array([0], dtype =int)
# hNGenTausType = np.array([0], dtype =int)


hMatchedTausPRes = np.array([0])
hMatchedTausPtRes = np.array([0])
hMatchedTausChargeRes = np.array([0])
hMatchedTausMaxAngleRes = np.array([0])
hMatchedTausNCompRes = np.array([0])
matched_tree.Branch("hMatchedTausPRes", hMatchedTausPRes, "hMatchedTausPRes/F")
matched_tree.Branch("hMatchedTausPtRes", hMatchedTausPtRes, "hMatchedTausPtRes/F")
matched_tree.Branch("hMatchedTausChargeRes", hMatchedTausChargeRes, "hMatchedTausChargeRes/F")
matched_tree.Branch("hMatchedTausMaxAngleRes", hMatchedTausMaxAngleRes, "hMatchedTausMaxAngleRes/F")
matched_tree.Branch("hMatchedTausNCompRes", hMatchedTausNCompRes, "hMatchedTausNCompRes/F")

# for branch in gen_const_tree.GetListOfBranches():
#     branch.SetBasketSize(16000)  # Reducir a la mitad
# for branch in reco_const_tree.GetListOfBranches():
#     branch.SetBasketSize(16000)  # Reducir a la mitad
# for branch in photon_tree.GetListOfBranches():
#     branch.SetBasketSize(16000)  # Reducir a la mitad
# for branch in matched_tree.GetListOfBranches():
#     branch.SetBasketSize(16000)  # Reducir a la mitad
# for branch in gen_tree.GetListOfBranches():
#     branch.SetBasketSize(16000)  # Reducir a la mitad

true_predicted_label = {"GenID": [], "True": [], "Predicted": [], "PhotonPredicted": []}
unmatched_true_label = {}
countEvents = 0
# run over all events
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
        hGenTauPt[0] =  genTauP4.Pt()  # Transverse momentum
        hGenVisTauPt[0] =genVisTauP4.Pt()  # Visible transverse momentum
        hGenTauP[0] = genTauP4.P()  # Momentum
        hGenVisTauP[0] = genVisTauP4.P()  # Visible momentum
        hGenVisTauMass[0] = genVisTauP4.M()  # Visible mass
        hGenTauType[0] =genTauId  # Tau decay type
        hGenTauQ[0] = genTauQ # Tau charge

        hGenTauEta[0] = genTauP4.Eta()  # Pseudo-rapidity
        hGenTauTheta[0] = genTauP4.Theta()  # Theta angle

        hGenTauDR[0] = genTauDR  # Angle of Tau Constituents
        hGenTauNConsts[0] = genTauNConsts  # Number of Tau Constituents

        gen_tree.Fill()
        
        countPionsRun = 0

        # print ("all GEN")
        # Look inside the generator level tau: check the constituents (decay products)

        # For each generator level tau, check the constituents and fill histograms
        for c in range(0, genTauNConsts):
            const = genTauConsts[c]
            constP4 = ROOT.TLorentzVector()
            constP4.SetXYZM(
                const.getMomentum().x,
                const.getMomentum().y,
                const.getMomentum().z,
                const.getMass(),
            )
            hGenConstPi0P[0] = -1
            hGenConstPi0Mass[0] = -1
            hGenConstPionP[0] = -1
            hGenConstPionPOverTauP[0] = -1
            hGenConstPion1P[0] = -1
            hGenConstPion2P[0] = -1
            hGenConstPion3P[0] = -1
            
            # Fill histograms
            hGenConstP[0]=constP4.P()  # Constituent momentum
            

            # Filling values for Pi0s
            # PDG ID for Pi0s == 111
            if const.getPDG() == 111:
                hGenConstPi0P[0] = constP4.P()
                hGenConstPi0Mass[0] = constP4.M()

                
                # No saved values
                # # Pi0s decay to Photons
                # daughtersPi0 = const.getDaughters()
                # # For each photon in the decay of a Pi0
                # for dauPhoton in daughtersPi0:
                #     dauPhotonP4 = ROOT.TLorentzVector()
                #     dauPhotonP4.SetXYZM(
                #         dauPhoton.getMomentum().x,
                #         dauPhoton.getMomentum().y,
                #         dauPhoton.getMomentum().z,
                #         dauPhoton.getMass(),
                #     )
                    # hGenConstPhotonP = np.append(hGenConstPhotonP, [dauPhotonP4.P()])
                    # hGenConstPhotonPOverTauP = np.append(hGenConstPhotonPOverTauP, [dauPhotonP4.P() / genVisTauP4.P()])


            # Filling values for Pions (charged)
            elif abs(const.getPDG()) == 211:  # PDG ID for charged pions
                hGenConstPionP[0] = constP4.P()
                hGenConstPionPOverTauP[0] = constP4.P() / genVisTauP4.P()

                # Counting the number of pions (three options)
                if countPionsRun == 0:
                    hGenConstPion1P[0] = constP4.P()
                    countPionsRun += 1
                elif countPionsRun == 1:
                    hGenConstPion2P[0] = constP4.P()
                    countPionsRun += 1
                elif countPionsRun == 2:
                    hGenConstPion3P[0] = constP4.P()
                    countPionsRun += 1
        # Fill gen constituents tree
        gen_const_tree.Fill()
        
            
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

        true_predicted_label["Predicted"].append(recoDM)
        true_predicted_label["PhotonPredicted"].append(recoTauId)

        hMatchedTausPRes[0] = (recoTauP4.P() - genTauP4.P()) / genTauP4.P()
        hMatchedTausPtRes[0] = (recoTauP4.Pt() - genTauP4.Pt()) / genTauP4.Pt()
        hMatchedTausChargeRes[0] = abs(recoTauQ) - abs(genTauQ) / (abs(genTauQ)+1e-6)
        hMatchedTausMaxAngleRes[0] = (recoTauDR - genTauDR) / genTauDR
        hMatchedTausNCompRes[0] = (recoTauNConsts - genTauNConsts) / genTauNConsts


        #          print ("Reco?",recoTauP4.P(),recoTauId,recoTauQ,recoTauDR,recoTauNConsts)

        # Now that we have a matched (gen,reco) pair, more checks for efficiency and resolution

        countPionsRun = 0
        # print ("Matched GEN!")
        # GEN: Look inside the tau, constituents:

        # Filling histograms for the matched tau with gen level information
        for c in range(0, genTauNConsts):
            const = genTauConsts[c]
            constP4 = ROOT.TLorentzVector()
            constP4.SetXYZM(
                const.getMomentum().x,
                const.getMomentum().y,
                const.getMomentum().z,
                const.getMass(),
            )

            hMatchedGenConstP[0] = constP4.P()
            hMatchedGenConstPOverTauP[0] = constP4.P()
            
            hMatchedGenConstPi0P[0] = -1
            hMatchedGenConstPi0Mass[0] = -1
            hMatchedGenConstPi0POverTauP[0] = -1
            
            hMatchedGenConstPionPOverTauP[0] = -1
            hMatchedGenConstPionP[0] = -1
            hMatchedGenConstPion1P[0] = -1
            hMatchedGenConstPion2P[0] = -1
            hMatchedGenConstPion3P[0] = -1


            if const.getPDG() == 111:
                hMatchedGenConstPi0P[0] = constP4.P()
                hMatchedGenConstPi0Mass[0] = constP4.M()
                hMatchedGenConstPi0POverTauP[0] = constP4.P() / genVisTauP4.P()

                # daughtersPi0 = const.getDaughters()
                # for dauPhoton in daughtersPi0:
                #     dauPhotonP4 = ROOT.TLorentzVector()
                #     dauPhotonP4.SetXYZM(
                #         dauPhoton.getMomentum().x,
                #         dauPhoton.getMomentum().y,
                #         dauPhoton.getMomentum().z,
                #         dauPhoton.getMass(),
                #     )
                #     hMatchedGenConstPhotonP = np.append(hMatchedGenConstPhotonP, [dauPhotonP4.P()])
                #     hMatchedGenConstPhotonPOverTauP = np.append(hMatchedGenConstPhotonPOverTauP, [dauPhotonP4.P() / genVisTauP4.P()])


            elif abs(const.getPDG()) == 211:
                hMatchedGenConstPionP[0] =constP4.P()
                hMatchedGenConstPionPOverTauP[0] = constP4.P() / genVisTauP4.P()
                if countPionsRun == 0:
                    hMatchedGenConstPion1P[0] = constP4.P()
                    countPionsRun += 1
                elif countPionsRun == 1:
                    hMatchedGenConstPion2P[0] = constP4.P()
                    countPionsRun += 1
                elif countPionsRun == 2:
                    hMatchedGenConstPion3P[0] = constP4.P()
                    countPionsRun += 1

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
            constP4.SetXYZM(
                const.getMomentum().x,
                const.getMomentum().y,
                const.getMomentum().z,
                const.getMass(),
            )
            hRecoConstP[0] = constP4.P()
            hRecoConstPOverTauP[0] = constP4.P() / recoTauP4.P()
            hRecoConstPhotonP[0] = -1
            hRecoConstPhotonPOverTauP[0] = -1
            hRecoConstPionP[0] = -1
            hRecoConstPionPOverTauP[0] = -1
            hRecoConstPion1P[0] = -1
            hRecoConstPion2P[0] = -1
            hRecoConstPion3P[0] = -1

            # This may be an error as we are confusing photons with neutrons

            logger_pi0mass.debug(f"Evaluating const to get photon with PDGID {const.getPDG()} and charge {const.getCharge()}")
            if const.getCharge() == 0:
                hRecoConstPhotonP[0] = constP4.P()
                hRecoConstPhotonPOverTauP[0] = constP4.P() / recoTauP4.P()
                photonCumulativeP4 += constP4
                
                recoTaus_photons[n_photons] = const
                n_photons += 1
            else:
                hRecoConstPionP[0] = constP4.P()
                hRecoConstPionPOverTauP[0] = constP4.P() / recoTauP4.P()
                if countPionsRun == 0:
                    hRecoConstPion1P[0] = constP4.P()
                    countPionsRun += 1
                elif countPionsRun == 1:
                    hRecoConstPion2P[0] = constP4.P()
                    countPionsRun += 1
                elif countPionsRun == 2:
                    hRecoConstPion3P[0] = constP4.P()
                    countPionsRun += 1
        # Fill reco constituents tree
        reco_const_tree.Fill()
        
        
        hRecoConstPi0MassFromPhotonMasstr[0] = -1
        hRecoConstlessPhotonPa1strMass[0] = -1
        hRecoConstxtraPhotonPrhostrMass[0] = -1
        hRecoThreePhotonMatchOnestrMassP[0] = -1
        hRecoThreePhotonMatchTwostrMassP[0] = -1
        hRecoThreePhotonNoMatchstrMassP[0] = -1
        hRecoConstPi0MassFromPhotonDiststr[0] = -1
        hRecoConstlessPhotonPa1strDist[0] = -1
        hRecoConstxtraPhotonPrhostrDist[0] = -1
        hRecoThreePhotonMatchOnestrDistP[0] = -1
        hRecoThreePhotonMatchTwostrDistP[0] = -1
        hRecoThreePhotonNoMatchstrDistP[0] = -1
        hRecoConstxtraPhotonPi[0] = -1
        hRecoConstlessPhotonPrho[0] = -1
        hRecoConstPi0Mass[0] = -1
        hRecoConstTwoPhotonAngDist[0] = -1
        if n_pi0s > 0:
            logger_pi0mass.debug(
                f"Found {n_pi0s} pi0s ({n_photons} photons) in the matched reco with Id {recoTauId} tau with real Id {genTauId}"
            )
            if n_photons == 3 and (genTauId == 2 or genTauId == 1):
                logger_pi0mass.debug(
                    f"Found 3 photons in the matched reco tau with real Id {genTauId}"
                )
                pi0Mass_strmass, noMatchedPhotons = pi0Reco.getPi0Mass(recoTaus_photons, strategy = {"mass":-1})
                if pi0Mass_strmass:
                    hRecoConstPi0MassFromPhotonMasstr[0] = pi0Mass_strmass
                
                if noMatchedPhotons:
                    for ide, photon in noMatchedPhotons.items(): 
                        PhotonP4 = ROOT.TLorentzVector()
                        PhotonP4.SetXYZM(
                        photon.getMomentum().x,
                        photon.getMomentum().y,
                        photon.getMomentum().z,
                        photon.getMass(),
                        )
                        if genTauId == 2:
                            hRecoConstlessPhotonPa1strMass[0] =PhotonP4.P()
                        elif genTauId == 1:
                            hRecoConstxtraPhotonPrhostrMass[0] = PhotonP4.P()
                    matched_keys = [key for key in range(3) if key not in noMatchedPhotons.keys()]
                    first_matched_P = ROOT.TLorentzVector()
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
                    non_matched_P = PhotonP4
                    hRecoThreePhotonMatchOnestrMassP[0] = first_matched_P.P()
                    hRecoThreePhotonMatchTwostrMassP[0] = second_matched_P.P()
                    hRecoThreePhotonNoMatchstrMassP[0] = non_matched_P.P()

                
                pi0Mass_strdist, noMatchedPhoton = pi0Reco.getPi0Mass(recoTaus_photons, strategy = {"distance":-1})
                if pi0Mass_strdist:
                    hRecoConstPi0MassFromPhotonDiststr[0] = pi0Mass_strdist
                if noMatchedPhoton:
                    for ide, photon in noMatchedPhotons.items(): 
                        PhotonP4 = ROOT.TLorentzVector()
                        PhotonP4.SetXYZM(
                        photon.getMomentum().x,
                        photon.getMomentum().y,
                        photon.getMomentum().z,
                        photon.getMass(),
                        )
                        if genTauId == 2:
                            hRecoConstlessPhotonPa1strDist[0] = PhotonP4.P()
                        elif genTauId == 1:
                            hRecoConstxtraPhotonPrhostrDist[0] = PhotonP4.P()
                    matched_keys = [key for key in range(3) if key not in noMatchedPhotons.keys()]
                    first_matched_P = ROOT.TLorentzVector()
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
                    non_matched_P = PhotonP4
                    hRecoThreePhotonMatchOnestrDistP[0] = first_matched_P.P()
                    hRecoThreePhotonMatchTwostrDistP[0] = second_matched_P.P()
                    hRecoThreePhotonNoMatchstrDistP[0] = non_matched_P.P()

                            
            elif n_photons == 1 and (genTauId == 0 or genTauId == 1):
                logger_pi0mass.debug(
                    f"Found 1 photons in the matched reco tau with real Id {genTauId}"
                )
                if genTauId == 0:
                    hRecoConstxtraPhotonPi[0] = photonCumulativeP4.P()
                elif genTauId == 1:
                    hRecoConstlessPhotonPrho[0] = photonCumulativeP4.P()
            
            elif n_photons == 2 and (genTauId == 1):
                logger_pi0mass.debug(
                    f"Found 2 photons in the matched reco tau with real Id {genTauId}"
                )
                hRecoConstPi0Mass[0] = photonCumulativeP4.M()
                photon1_P4 = ROOT.TLorentzVector()
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
                ang_dist = myutils.dRAngle(
                    photon1_P4, photon2_P4
                )
                hRecoConstTwoPhotonAngDist[0] = ang_dist
        # Fill photon constituents tree    
        photon_tree.Fill()

        hRecoTauType[0] = recoTauId
        hRecoTauPt[0] = recoTauP4.Pt()
        hRecoTauP[0] = recoTauP4.P()
        hRecoTauMass[0] = recoTauP4.M()
        hRecoTauQ[0] = recoTauQ
        hRecoTauEta[0] = recoTauP4.Eta()
        hRecoTauTheta[0] = recoTauP4.Theta()

        hRecoTauDR[0] = recoTauDR
        hMatchedGenTauPt[0] = genTauP4.Pt()
        hMatchedGenTauP[0] = genTauP4.P()
        hMatchedGenTauType[0] = genTauId
        hMatchedGenTauQ[0] = genTauQ
        hMatchedGenTauEta[0] = genTauP4.Eta()
        hMatchedGenTauTheta[0] = genTauP4.Theta()

        hMatchedGenTauDR[0] = genTauDR

        # Resolution plots:
        hResTauPt = np.append(hResTauPt, [(recoTauP4.Pt() - genVisTauP4.Pt()) / (genVisTauP4.Pt()+1e-6)])
        hResTauMass = np.append(hResTauMass, [(recoTauP4.M() - genVisTauP4.M()) / (genVisTauP4.M()+1e-6)])
        hResTauP = np.append(hResTauP, [(recoTauP4.P() - genVisTauP4.P()) / (genVisTauP4.P()+1e-6)])

        matched_tree.Fill()
        

    # # print ("Taus???",nGenTaus,nTaus)
    # hNTaus = np.append(hNTaus, [nRecoTaus])
    # hNTausType = np.append(hNTausType, [nTausType])
    # hNGenTausType = np.append(hNGenTausType, [nGenTausType])
    # hNGenTaus = np.append(hNGenTaus, [nGenTausHad])



# Do efficiencies (divide matched gen by all gen)
# Evita división por cero con np.where
# hEffiGenPi0Mass = np.where(hGenConstPi0Mass != 0, hMatchedGenConstPi0Mass / hGenConstPi0Mass, 0)
# hEffiGenTauPt = np.where(hGenTauPt != 0, hMatchedGenTauPt / hGenTauPt, 0)
# hEffiGenVisTauPt = np.where(hGenVisTauPt != 0, hMatchedGenVisTauPt / hGenVisTauPt, 0)
# hEffiGenTauP = np.where(hGenTauP != 0, hMatchedGenTauP / hGenTauP, 0)
# hEffiGenVisTauP = np.where(hGenVisTauP != 0, hMatchedGenVisTauP / hGenVisTauP, 0)
# hEffiGenVisTauMass = np.where(hGenVisTauMass != 0, hMatchedGenVisTauMass / hGenVisTauMass, 0)
# hEffiGenTauEta = np.where(hGenTauEta != 0, hMatchedGenTauEta / hGenTauEta, 0)
# hEffiGenTauTheta = np.where(hGenTauTheta != 0, hMatchedGenTauTheta / hGenTauTheta, 0)
# hEffiGenTauType = np.where(hGenTauType != 0, hMatchedGenTauType / hGenTauType, 0)

logger_process.info("Found %d events", countEvents)

# decaystr = "decayAll" if selectDecay == -777 else "decay{}".format(selectDecay)
# true_predicted_label_output_file = outputpath + f"true_predicted_label_{decaystr}.csv"
# true_predicted_label_df = pd.DataFrame(true_predicted_label)
# true_predicted_label_df.to_csv(true_predicted_label_output_file, index=False)

# # Check if config["output"]["outputlabels"] is a list
# if type(config["output"]["outputlabels"]) is not list:
#     if config["output"]["outputlabels"] is None:
#         config["output"]["outputlabels"] = []
#     else:
#         config["output"]["outputlabels"] = [config["output"]["outputlabels"]]
# if true_predicted_label_output_file not in config["output"]["outputlabels"]:
#     config["output"]["outputlabels"].append(true_predicted_label_output_file)

output_config_file = outputpath + "config.yaml"
with open(output_config_file, "w") as file:
    yaml.dump(config, file)
    logger_io.info("Configuration file saved to %s", output_config_file)

outfile = ROOT.TFile(outputpath + "Trees_"+fileOutName, "RECREATE")
outfile.cd()

# Abrir el archivo de salida
# outfile = TFile("output.root", "RECREATE")
# Guardamos los árboles en el archivo de salida
print("N Entries Gen Tree", gen_tree.GetEntries())
print("Variables in Gen Tree", gen_tree.Print())
print("N Entries Reco Tree", reco_const_tree.GetEntries())
print("Variables in Reco Tree", reco_const_tree.Print())
print("N Entries Photon Tree", photon_tree.GetEntries())
print("Variables in Photon Tree", photon_tree.Print())
print("N Entries Matched Tree", matched_tree.GetEntries())
print("Variables in Matched Tree", matched_tree.Print())
print("N Entries Gen Const Tree", gen_const_tree.GetEntries())
print("Variables in Gen Const Tree", gen_const_tree.Print())


gen_tree.SetDirectory(outfile)
reco_const_tree.SetDirectory(outfile)
photon_tree.SetDirectory(outfile)
matched_tree.SetDirectory(outfile)
gen_const_tree.SetDirectory(outfile)
gen_tree.Write("gen_tree")
reco_const_tree.Write("reco_const_tree")
photon_tree.Write("photon_tree")
matched_tree.Write("matched_tree")
gen_const_tree.Write("gen_const_tree")


logger_io.info("Output file %s", outputpath + fileOutName)
logger_io.info("End of job")
outfile.Close()