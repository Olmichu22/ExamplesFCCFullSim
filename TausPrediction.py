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

from modules import pi0Reco
from modules import tauReco, electronReco, muonReco
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

config["cuts"]["generalPCut"] = (
    args.generalPCut if args.generalPCut != None else config["cuts"]["generalPCut"]
)
dRMax = config["cuts"]["dRMax"]
minPTauPhoton = config["cuts"]["TauPhotonPCut"]
minPTauPion = config["cuts"]["TauPionPCut"]
PNeutron = config["cuts"]["NeutronCut"]
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
outputbasepath = "Results/TauPredictions/"


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

config["general"]["test"] = (
    args.test if args.test != None else config["general"]["test"]
)

sample = config["general"]["sample"]

# Check if sample is a list
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
    nfiles = 2

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
true_predicted_label = {"GenID": [], "True": [], "Predicted": [], "PhotonPredicted": []}
unmatched_true_label = {}
countEvents = 0

result_labels = {}
result_labels["tau1"] = []
result_labels["tau2"] = []
result_labels["id-tau1"] = []
result_labels["id-tau2"] = []

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
    recoElectrons = electronReco.findAllElectrons(pfos, generalPCut)
    recoMuons = muonReco.findAllMuons(pfos, generalPCut)
    
    nRecoTaus = len(recoTaus)
    nRecoElectrons = len(recoElectrons)
    nRecoMuons = len(recoMuons)
    
    recoParticles = {}
    pidx = 0
    for taui in range(nRecoTaus):
        recoParticles[pidx] = recoTaus[taui]
        pidx += 1
    for elei in range(nRecoElectrons):
        recoParticles[pidx] = recoElectrons[elei]
        pidx += 1
    for mui in range(nRecoMuons):
        recoParticles[pidx] = recoMuons[mui]
        pidx += 1
        
    nRecoParticles = len(recoParticles)

    logger_process.debug(
        "Found %d reconstructed taus. Details:\n%s",
        nRecoParticles,
        "\n".join("RecoTau %d: %s" % (i, tau) for i, tau in recoParticles.items()),
    )
    
    if nGenTaus == 0 or nGenTaus > 2:
        logger_process.debug("No gen taus or more than 2 gen taus found")
        result_labels["tau1"].append(-999)
        result_labels["tau2"].append(-999)
        result_labels["id-tau1"].append(-999)
        result_labels["id-tau2"].append(-999)
        continue

    gen_tau_ids = []
    for i in range(0, nGenTaus):
        gen_tau_ids.append(genTaus[i].getID())
    
    reco_tau_ids = []
    for i in range(0, nRecoParticles):
        reco_tau_ids.append(recoParticles[i].getID())
    
    if len(reco_tau_ids) < 2:
        while len(reco_tau_ids) < 2:
            reco_tau_ids.append(-1)
    elif len(reco_tau_ids) > 2:
        reco_tau_ids = [-1, -1]
            
    if len(gen_tau_ids) < 2:
        while len(gen_tau_ids) < 2:
            gen_tau_ids.append(-1)
    
    if set(reco_tau_ids) != set([-1]):
        if -1 in reco_tau_ids:
            key, cos_angle = myutils.associate_reco_with_gen_taus(genTaus, recoParticles[0])
            key_order = [key, len(genTaus) - 1 - key]
        else:
            key_1, cos_angle_1 = myutils.associate_reco_with_gen_taus(genTaus, recoParticles[0])
            key_2, cos_angle_2 = myutils.associate_reco_with_gen_taus(genTaus, recoParticles[1])
            if key_1 == key_2:
                if cos_angle_1 > cos_angle_2:
                    key_order = [key_1, 1-key_2]
                else:
                    key_order = [key_2, 1-key_1]
            else:
                key_order = [key_1, key_2]
    else:
        key_order = [0, 1]
        
    if key_order[0] == key_order[1]:
        print(key_order[0], key_order[1])
        print(reco_tau_ids)
        print(gen_tau_ids)
        print(eventid)
    
    for i in range(2):
        
        result_labels[f"tau{i+1}"].append(gen_tau_ids[i])
        
        recoTauId = reco_tau_ids[i]
        if recoTauId > 0:
            if recoTauId < 10 and recoTauId >= 0:
                nPhotons = recoTauId
                n_pi0s = math.ceil(nPhotons / 2)
                recoDM = n_pi0s
            elif recoTauId >= 10:
                nPhotons = recoTauId - 10
                n_pi0s = math.ceil(nPhotons / 2)
                recoDM = 10 + n_pi0s
        elif recoTauId == -1:
            recoDM = -1
        else:
            recoDM = recoTauId
        
        result_labels[f"id-tau{key_order[i]+1}"].append(recoDM)
    
    # print(len(result_labels["tau1"]), len(result_labels["tau2"]),
    #       len(result_labels["id-tau1"]), len(result_labels["id-tau2"]))

logger_process.info("Predicted %d events", countEvents)

decaystr = "decayAll" if selectDecay == -777 else "decay{}".format(selectDecay)

result_labels = pd.DataFrame(result_labels)
result_labels.to_csv(outputpath + "result_labels_pfo_" +decaystr+".csv", index=False)


output_config_file = outputpath + "config.yaml"
with open(output_config_file, "w") as file:
    yaml.dump(config, file)
    logger_io.info("Configuration file saved to %s", output_config_file)

logger_io.info("Output file %s", outputpath + fileOutName)
logger_io.info("End of job")
