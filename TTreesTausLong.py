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
path = "/pnfs/ciemat.es/data/cms/store/user/cepeda/FCC/FullSim/"
file = "out_reco_edm4hep_edm4hep"
filenames = []
dir_path = path + "/" + sample

nfiles = len(os.listdir(dir_path))

nfiles = 1000
if test == True:
    nfiles = 5

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
outfile = ROOT.TFile(outputpath + "Trees_"+fileOutName, "RECREATE")
Tau_tree = TTree("Tau_tree", f"Tree {cut_string}_Sample_{sample}")

# Defining many histogram
GenEventId = np.array([0], dtype=int)
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

Tau_tree.Branch("GenEventId", GenEventId, "GenEventId/I")
Tau_tree.Branch("hGenTauPt", hGenTauPt, "hGenTauPt/F")
Tau_tree.Branch("hGenVisTauPt", hGenVisTauPt, "hGenVisTauPt/F")
Tau_tree.Branch("hGenTauP", hGenTauP, "hGenTauP/F")
Tau_tree.Branch("hGenVisTauP", hGenVisTauP, "hGenVisTauP/F")
Tau_tree.Branch("hGenTauType", hGenTauType, "hGenTauType/I")
Tau_tree.Branch("hGenVisTauMass", hGenVisTauMass, "hGenVisTauMass/F")
Tau_tree.Branch("hGenTauQ", hGenTauQ, "hGenTauQ/F")
Tau_tree.Branch("hGenTauEta", hGenTauEta, "hGenTauEta/F")
Tau_tree.Branch("hGenTauTheta", hGenTauTheta, "hGenTauTheta/F")
Tau_tree.Branch("hGenTauDR", hGenTauDR, "hGenTauDR/F")

IsMatched = np.array([0], dtype =int)
hMatchedGenTauPt = ROOT.std.vector('float')()
hMatchedGenVisTauPt = ROOT.std.vector('float')()
hMatchedGenTauP = ROOT.std.vector('float')()
hMatchedGenVisTauP = ROOT.std.vector('float')()
Tau_tree.Branch("IsMatched", IsMatched, "IsMatched/I")
Tau_tree.Branch("hMatchedGenTauPt", hMatchedGenTauPt)
Tau_tree.Branch("hMatchedGenVisTauPt", hMatchedGenVisTauPt)
Tau_tree.Branch("hMatchedGenTauP", hMatchedGenTauP)
Tau_tree.Branch("hMatchedGenVisTauP", hMatchedGenVisTauP)

hMatchedGenTauType = np.array([-1], dtype =int)
hMatchedGenVisTauMass = ROOT.std.vector('float')()
hMatchedGenTauQ = ROOT.std.vector('float')()
hMatchedGenTauEta = ROOT.std.vector('float')()
hMatchedGenTauTheta = ROOT.std.vector('float')()
hMatchedGenTauDR = ROOT.std.vector('float')()
Tau_tree.Branch("hMatchedGenTauType", hMatchedGenTauType, "hMatchedGenTauType/I")
Tau_tree.Branch("hMatchedGenVisTauMass", hMatchedGenVisTauMass)
Tau_tree.Branch("hMatchedGenTauQ", hMatchedGenTauQ)
Tau_tree.Branch("hMatchedGenTauEta", hMatchedGenTauEta)
Tau_tree.Branch("hMatchedGenTauTheta", hMatchedGenTauTheta)
Tau_tree.Branch("hMatchedGenTauDR", hMatchedGenTauDR)


hRecoTauPt = ROOT.std.vector('float')()
hRecoTauP = ROOT.std.vector('float')()
Tau_tree.Branch("hRecoTauPt", hRecoTauPt)
Tau_tree.Branch("hRecoTauP", hRecoTauP)

hRecoTauMass = ROOT.std.vector('float')()
hRecoTauType = np.array([-1], dtype =int)
hRecoTauDM = np.array([-1], dtype =int)
hRecoTauQ = ROOT.std.vector('float')()
hRecoTauEta = ROOT.std.vector('float')()
hRecoTauTheta = ROOT.std.vector('float')()
hRecoTauDR = ROOT.std.vector('float')()
Tau_tree.Branch("hRecoTauMass", hRecoTauMass)
Tau_tree.Branch("hRecoTauType", hRecoTauType, "hRecoTauType/I")
Tau_tree.Branch("hRecoTauDM", hRecoTauDM, "hRecoTauDM/I")
Tau_tree.Branch("hRecoTauQ", hRecoTauQ)
Tau_tree.Branch("hRecoTauEta", hRecoTauEta)
Tau_tree.Branch("hRecoTauTheta", hRecoTauTheta)
Tau_tree.Branch("hRecoTauDR", hRecoTauDR)

nRecoPhotons = np.array([0], dtype =int)
hRecoConstPhotonP = ROOT.std.vector('float')()
nRecoPions = np.array([0], dtype =int)
hRecoConstPionP = ROOT.std.vector('float')()
Tau_tree.Branch("nRecoPhotons", nRecoPhotons, "nRecoPhotons/I")
Tau_tree.Branch("nRecoPions", nRecoPions, "nRecoPions/I")
Tau_tree.Branch("hRecoConstPhotonP", hRecoConstPhotonP)
Tau_tree.Branch("hRecoConstPionP", hRecoConstPionP)

nRecoPions0 = np.array([0], dtype =int)
hRecoConstPi0Mass = ROOT.std.vector('float')()
hRecoConstTwoPhotonAngDist = ROOT.std.vector('float')()
Tau_tree.Branch("nRecoPions0", nRecoPions0, "nRecoPions0/I")
Tau_tree.Branch("hRecoConstPi0Mass", hRecoConstPi0Mass)
Tau_tree.Branch("hRecoConstTwoPhotonAngDist", hRecoConstTwoPhotonAngDist)


hRecoConstPi0MassFromPhotonMasstr = ROOT.std.vector('float')()
hRecoConstPi0MassFromPhotonDiststr = ROOT.std.vector('float')()
Tau_tree.Branch("hRecoConstPi0MassFromPhotonMasstr", hRecoConstPi0MassFromPhotonMasstr, "hRecoConstPi0MassFromPhotonMasstr/F")
Tau_tree.Branch("hRecoConstPi0MassFromPhotonDiststr", hRecoConstPi0MassFromPhotonDiststr, "hRecoConstPi0MassFromPhotonDiststr/F")

nGenPions0 = np.array([0], dtype =int)
hGenConstPi0Mass = ROOT.std.vector('float')()
Tau_tree.Branch("nGenPions0", nGenPions0, "nGenPions0/I")
Tau_tree.Branch("hGenConstPi0Mass", hGenConstPi0Mass)

nGenPions = np.array([0], dtype =int)
hGenConstPi0P = ROOT.std.vector('float')()
hGenConstPionP = ROOT.std.vector('float')()
# hGenConstPhotonP = np.array([0])
Tau_tree.Branch("nGenPions", nGenPions, "nGenPions/I")
Tau_tree.Branch("hGenConstPi0P", hGenConstPi0P)
Tau_tree.Branch("hGenConstPionP", hGenConstPionP)
# gen_const_tree.Branch("hGenConstPhotonP", hGenConstPhotonP, "hGenConstPhotonP/F")

# Hist for reconstructed photons in a1 (pi pi0 pi0), rho (pi0 pi0), and pi cases
# 3 photon cases
hRecoConstlessPhotonPa1strMass = ROOT.std.vector('float')()
hRecoConstlessPhotonPa1strDist = ROOT.std.vector('float')()
Tau_tree.Branch("hRecoConstlessPhotonPa1strMass", hRecoConstlessPhotonPa1strMass)
Tau_tree.Branch("hRecoConstlessPhotonPa1strDist", hRecoConstlessPhotonPa1strDist)

hRecoConstxtraPhotonPrhostrMass = ROOT.std.vector('float')()
hRecoConstxtraPhotonPrhostrDist = ROOT.std.vector('float')()
Tau_tree.Branch("hRecoConstxtraPhotonPrhostrMass", hRecoConstxtraPhotonPrhostrMass)
Tau_tree.Branch("hRecoConstxtraPhotonPrhostrDist", hRecoConstxtraPhotonPrhostrDist)

# 1 photon cases
hRecoConstlessPhotonPrho = ROOT.std.vector('float')()
hRecoConstxtraPhotonPi = ROOT.std.vector('float')()
Tau_tree.Branch("hRecoConstlessPhotonPrho", hRecoConstlessPhotonPrho)
Tau_tree.Branch("hRecoConstxtraPhotonPi", hRecoConstxtraPhotonPi)

hRecoThreePhotonMatchOnestrMassP = ROOT.std.vector('float')()
hRecoThreePhotonMatchOnestrDistP = ROOT.std.vector('float')()
hRecoThreePhotonMatchTwostrMassP = ROOT.std.vector('float')()
hRecoThreePhotonMatchTwostrDistP = ROOT.std.vector('float')()
hRecoThreePhotonNoMatchstrMassP = ROOT.std.vector('float')()
hRecoThreePhotonNoMatchstrDistP = ROOT.std.vector('float')()
Tau_tree.Branch("hRecoThreePhotonMatchOnestrMassP", hRecoThreePhotonMatchOnestrMassP)
Tau_tree.Branch("hRecoThreePhotonMatchOnestrDistP", hRecoThreePhotonMatchOnestrDistP)
Tau_tree.Branch("hRecoThreePhotonMatchTwostrMassP", hRecoThreePhotonMatchTwostrMassP)
Tau_tree.Branch("hRecoThreePhotonMatchTwostrDistP", hRecoThreePhotonMatchTwostrDistP)
Tau_tree.Branch("hRecoThreePhotonNoMatchstrMassP", hRecoThreePhotonNoMatchstrMassP)
Tau_tree.Branch("hRecoThreePhotonNoMatchstrDistP", hRecoThreePhotonNoMatchstrDistP)

nMatchedGenPions = np.array([0], dtype =int)
nMatchedGenPions0 = np.array([0], dtype =int)
hMatchedGenConstPionP = ROOT.std.vector('float')()
hMatchedGenConstPi0Mass = ROOT.std.vector('float')()
Tau_tree.Branch("hMatchedGenConstPi0Mass", hMatchedGenConstPi0Mass)
Tau_tree.Branch("nMatchedGenPions0", nMatchedGenPions0, "nMatchedGenPions0/I")
Tau_tree.Branch("hMatchedGenConstPionP", hMatchedGenConstPionP)
Tau_tree.Branch("nMatchedGenPions", nMatchedGenPions, "nMatchedGenPions/I")


hGenTauNConsts = np.array([0], dtype =int)
Tau_tree.Branch("hGenTauNConsts", hGenTauNConsts, "hGenTauNConsts/I")

vectors = [
    hMatchedGenTauPt,
    hMatchedGenVisTauPt,
    hMatchedGenTauP,
    hMatchedGenVisTauP,
    hMatchedGenVisTauMass,
    hMatchedGenTauQ,
    hMatchedGenTauEta,
    hMatchedGenTauTheta,
    hMatchedGenTauDR,
    hRecoTauPt,
    hRecoTauP,
    hRecoTauMass,
    hRecoTauQ,
    hRecoTauEta,
    hRecoTauTheta,
    hRecoTauDR,
    hRecoConstPhotonP,
    hRecoConstPionP,
    hRecoConstPi0Mass,
    hRecoConstTwoPhotonAngDist,
    hRecoConstPi0MassFromPhotonMasstr,
    hRecoConstPi0MassFromPhotonDiststr,
    hGenConstPi0Mass,
    hMatchedGenConstPi0Mass,
    hGenConstPi0P,
    hGenConstPionP,
    hRecoConstlessPhotonPa1strMass,
    hRecoConstlessPhotonPa1strDist,
    hRecoConstlessPhotonPrho,
    hRecoConstxtraPhotonPi,
    hRecoThreePhotonMatchOnestrMassP,
    hRecoThreePhotonMatchOnestrDistP,
    hRecoThreePhotonMatchTwostrMassP,
    hRecoThreePhotonMatchTwostrDistP,
    hRecoThreePhotonNoMatchstrMassP,
    hRecoThreePhotonNoMatchstrDistP,
    hMatchedGenConstPionP,
]

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
        for v in vectors:
            v.clear()
        nGenPions0[0] = 0
        nGenPions[0] = 0
        nRecoPhotons[0] = 0
        nRecoPions[0] = 0
        nMatchedGenPions0[0] = 0
        nMatchedGenPions[0] = 0
        nRecoPions0[0] = 0
        hRecoTauType[0] = -1
        hRecoTauDM[0] = -1
        hMatchedGenTauType[0] = -1
            
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
        GenEventId[0] = eventid  # Event ID
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
            

            # Filling values for Pi0s
            # PDG ID for Pi0s == 111
            if const.getPDG() == 111:
                nGenPions0[0] += 1
                hGenConstPi0P.push_back(constP4.P())
                hGenConstPi0Mass.push_back(constP4.M())

                
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
                nGenPions[0] += 1
                hGenConstPionP.push_back(constP4.P())
        
            
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
            # Fill the tree with the unmatched gen tau
            IsMatched[0] = 0
            Tau_tree.Fill()
            continue
        
        IsMatched[0] = 1
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



            if const.getPDG() == 111:
                nMatchedGenPions0[0] += 1
                hMatchedGenConstPi0Mass.push_back(constP4.M())

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
                nMatchedGenPions[0] += 1
                hMatchedGenConstPionP.push_back(constP4.P())
               

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

            # This may be an error as we are confusing photons with neutrons

            logger_pi0mass.debug(f"Evaluating const to get photon with PDGID {const.getPDG()} and charge {const.getCharge()}")
            if const.getCharge() == 0:
                nRecoPhotons[0] += 1
                hRecoConstPhotonP.push_back(constP4.P())
                photonCumulativeP4 += constP4
                
                recoTaus_photons[n_photons] = const
                n_photons += 1
            else:
                nRecoPions[0] += 1
                hRecoConstPionP.push_back(constP4.P())
                

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
                    hRecoConstPi0MassFromPhotonMasstr.push_back(pi0Mass_strmass)
                
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
                            hRecoConstlessPhotonPa1strMass.push_back(PhotonP4.P())
                        elif genTauId == 1:
                            hRecoConstxtraPhotonPrhostrMass.push_back(PhotonP4.P())
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
                    hRecoThreePhotonMatchOnestrMassP.push_back(first_matched_P.P())
                    hRecoThreePhotonMatchTwostrMassP.push_back(second_matched_P.P())
                    hRecoThreePhotonNoMatchstrMassP.push_back(non_matched_P.P())

                
                pi0Mass_strdist, noMatchedPhoton = pi0Reco.getPi0Mass(recoTaus_photons, strategy = {"distance":-1})
                if pi0Mass_strdist:
                    hRecoConstPi0MassFromPhotonDiststr.push_back(pi0Mass_strdist)
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
                            hRecoConstlessPhotonPa1strDist.push_back(PhotonP4.P())
                        elif genTauId == 1:
                            hRecoConstxtraPhotonPrhostrDist.push_back(PhotonP4.P())
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
                    hRecoThreePhotonMatchOnestrDistP.push_back(first_matched_P.P())
                    hRecoThreePhotonMatchTwostrDistP.push_back(second_matched_P.P())
                    hRecoThreePhotonNoMatchstrDistP.push_back(non_matched_P.P())

                            
            elif n_photons == 1 and (genTauId == 0 or genTauId == 1):
                logger_pi0mass.debug(
                    f"Found 1 photons in the matched reco tau with real Id {genTauId}"
                )
                if genTauId == 0:
                    hRecoConstxtraPhotonPi.push_back(photonCumulativeP4.P())
                elif genTauId == 1:
                    hRecoConstlessPhotonPrho.push_back(photonCumulativeP4.P())
            
            elif n_photons == 2 and (genTauId == 1):
                logger_pi0mass.debug(
                    f"Found 2 photons in the matched reco tau with real Id {genTauId}"
                )
                hRecoConstPi0Mass.push_back(photonCumulativeP4.M())
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
                hRecoConstTwoPhotonAngDist.push_back(ang_dist)
        # Fill photon constituents tree    


        hRecoTauType[0] = recoTauId
        hRecoTauDM[0] = recoDM
        
        hRecoTauPt.push_back(recoTauP4.Pt())
        hRecoTauP.push_back(recoTauP4.P())
        hRecoTauMass.push_back(recoTauP4.M())
        hRecoTauQ.push_back(recoTauQ)
        hRecoTauEta.push_back(recoTauP4.Eta())
        hRecoTauTheta.push_back(recoTauP4.Theta())

        hRecoTauDR.push_back(recoTauDR)
        hMatchedGenTauPt.push_back(genTauP4.Pt())
        hMatchedGenTauP.push_back(genTauP4.P())
        hMatchedGenTauType[0] = genTauId
        hMatchedGenTauQ.push_back(genTauQ)
        hMatchedGenTauEta.push_back(genTauP4.Eta())
        hMatchedGenTauTheta.push_back(genTauP4.Theta())

        hMatchedGenTauDR.push_back(genTauDR)

        # Fill the Tree
        Tau_tree.Fill()
        

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



# Abrir el archivo de salida
# outfile = TFile("output.root", "RECREATE")
# Guardamos los árboles en el archivo de salida



Tau_tree.SetDirectory(outfile)

Tau_tree.Write()

logger_io.info("Output file %s", outputpath + fileOutName)
logger_io.info("End of job")
outfile.Close()
