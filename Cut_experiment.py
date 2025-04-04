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
import logging
import copy

from modules import tauReco 
from modules import myutils 

import argparse
parser = argparse.ArgumentParser(description="Configure the analysis",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("-f", "--sample")
parser.add_argument("-o", "--outfile")
parser.add_argument("-d", "--decay", type=int)  # GEN
parser.add_argument(
    "-V",
    "--value",
    type=str,
    help="Value to modify. It has to be one of the available cuts: NeutronP, PhotonP, PionP, dRMax, MatchedGenMinDR, generalP",
)
parser.add_argument(
    "-p", "--TauPhotonPCut", type=float
)  

parser.add_argument("-i", "--TauPionPCut", nargs ="+", type=float)
parser.add_argument("-R", "--dRMax", nargs ="+", type=float)
parser.add_argument("-n", "--NeutronCut", nargs ="+", type=float)
parser.add_argument("-g", "--generalPCut", nargs ="+", type=float)
parser.add_argument("-r", "--MatchedGenMinDR", nargs ="+", type=float)

parser.add_argument("-l", "--range", nargs="+", type=float, help="Range of the value to modify: min, max, step")
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

outputbasepath = "Results/Experiments/"
values = ["NeutronCut", "TauPhotonPCut", "TauPionPCut", "dRMax", "MatchedGenMinDR", "generalPCut"]

# Load Config File
default_config = "config/default/cutexperiment.yaml"
config = myutils.load_yaml_config(args.config, default_config)

# Set User Config (if any)
config["experiment"] = args.value if args.value is not None else config["experiment"]

if config["experiment"] not in values:
    raise Exception(
        f"Value '{config['experiment']}' not valid. Choose between {values}"
    )


for key in config["cuts"]:
  if key == config["experiment"]:
    if args.range is not None and len(args.range) != 3:
        raise Exception(
            f"Invalid range format. Expected format: min, max, step. Got {len(args.range)} values."
        )
    if args.range is not None:    
      config["cuts"][key] = np.arange(args.range[0], args.range[1], args.range[2]).tolist()
    else:
      config["cuts"][key] = getattr(args, key) if getattr(args, key) is not None else config["cuts"][key]
  else:
    config["cuts"][key] = getattr(args, key)[0] if getattr(args, key) is not None else config["cuts"][key][0]


dRMax = config["cuts"]["dRMax"]
minPTauPhoton = config["cuts"]["TauPhotonPCut"]
minPTauPion = config["cuts"]["TauPionPCut"]
PNeutron = config["cuts"]["NeutronCut"]
dRMatch = config["cuts"]["MatchedGenMinDR"]
generalPCut = config["cuts"]["generalPCut"]


exp_str = ""
for key, value in config["cuts"].items():
    if key == config["experiment"]:
        exp_str += f"{key}Exp_"
    else:
        exp_str += f"{key}{value}_"
cut_string = f"_{exp_str[:-1]}"

config["general"]["decay"] = args.decay if args.decay != None else config["general"]["decay"]
selectDecay = config["general"]["decay"]

decayString = f"decay{selectDecay}" + cut_string
if selectDecay == -777:
    decayString = "decayAll" + cut_string

config["general"]["outfile"] = (
    args.outfile if args.outfile != None else config["general"]["outfile"]
)
outfile = config["general"]["outfile"]   


fileOutName = outfile + decayString + ".root"

outputpath = outputbasepath + outfile + cut_string + "/"
if not os.path.exists(outputpath):
    os.makedirs(outputpath)

# Once set the output path, we can set the logger
if args.verbose == 0:
    log_level = logging.WARNING  # Only warnings and errors
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(outputpath + "/" + "exp.log", mode="w"),
    ]
elif args.verbose == 1:
    log_level = logging.INFO  # Informational messages
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(outputpath + "/" + "exp.log", mode="w"),
    ]
else:
    log_level = logging.DEBUG  # Debug messages for -vv or higher
    handlers=[logging.FileHandler(outputpath + "/" + "exp.log", mode="w")]
logging.basicConfig(
    level=log_level,
    format="%(asctime)s, %(levelname)s, [%(name)s] - %(message)s",
    handlers=handlers)
logger_config = logging.getLogger("config")
logger_io = logging.getLogger("io")
logger_process = logging.getLogger("processing")

logger_config.info("Logging handler initialized.")

# Finish Configuration
config["general"]["sample"] = args.sample if args.sample!= None else config["general"]["sample"]
config["general"]["matchedCM"] = args.matchedCM if args.matchedCM != None else config["general"]["matchedCM"]
config["general"]["test"] = args.test if args.test != None else config["general"]["test"]

sample = config["general"]["sample"]
matched_cm_arg = config["general"]["matchedCM"]
matched_cm = True if matched_cm_arg=="True" else False
test_arg = config["general"]["test"]
test = True if test_arg=="True" else False


config["output"]["outputpath"] = outputpath
config["output"]["outputfile"] = fileOutName

logger_config.info("Configuration loaded!")
logger_config.info("Configuration:\n%s", pprint.pformat(config, indent=4))


# get all the files
path="/pnfs/ciemat.es/data/cms/store/user/cepeda/FCC/FullSim/"
file="out_reco_edm4hep_edm4hep"
filenames=[]
dir_path=path+"/"+sample
names = ROOT.std.vector('string')()
nfiles=len(os.listdir(dir_path))

nfiles=1000
if test==True:
   nfiles=10

logger_io.info("Reading files from %s", dir_path)
for i in range(1,nfiles+1):
    filename=dir_path+"/"+file+"_{}.root".format(i)
    logger_io.debug("Opening file %s", filename)
    my_file = Path(filename)
    if my_file.is_file():
        root_file = myutils.open_root_file(filename)
        if not root_file or root_file.IsZombie():
            logger_io.warning("File %s is a zombie or could not be opened.", filename)
            continue
        filenames.append(filename)



true_predicted_label_sample = {"GenID":[],"True":[], "Predicted":[], "PhotonPredicted":[]}


# collections to use
genparts = "MCParticles"
pfobjects="PandoraPFOs"


experiment_parameters={}
for key, value in config["cuts"].items():
  if key == config["experiment"]:
    experiment_parameters[key] = None
  else:
    experiment_parameters[key] = value

for exp, exp_value in enumerate(config["cuts"][config["experiment"]]):
    experiment_parameters[config["experiment"]] = exp_value
    logger_process.info("Running experiment with parameters: %s", experiment_parameters)
    true_predicted_label = copy.deepcopy(true_predicted_label_sample)

    logger_io.info("Found %d files", len(filenames))
    reader = root_io.Reader(filenames)
    countEvents=0
    
    for eventid, event in enumerate(reader.get("events")):
        logger_process.debug("Processing event %d", eventid)
        if countEvents%1000==0:
            logger_process.info("Processing event %d of experiment %d out of %d", countEvents, exp+1, len(config["cuts"][config["experiment"]]))
        countEvents+=1

        mc_particles = event.get(genparts)
        pfos = event.get(pfobjects)

        genTaus=tauReco.findAllGenTaus(mc_particles)
        nGenTaus=len(genTaus)
        logger_process.debug(
        "Found %d gen taus. Details:\n%s",
        nGenTaus,
        "\n".join("GenTau %d: %s" % (i, tau) for i, tau in genTaus.items()),
        )
        logger_process.debug("Running tau reconstruction with parameters: %s", experiment_parameters)
        recoTaus = tauReco.findAllTaus(pfos,
                                   experiment_parameters["dRMax"],
                                   experiment_parameters["TauPhotonPCut"],
                                   experiment_parameters["TauPionPCut"],
                                   experiment_parameters["NeutronCut"],
                                   experiment_parameters["generalPCut"])

        nRecoTaus=len(recoTaus)
        logger_process.debug(
          "Found %d reconstructed taus. Details:\n%s",
          nRecoTaus,
          "\n".join("RecoTau %d: %s" % (i, tau) for i, tau in recoTaus.items()),
            )

        nTausType = 0
        for i in range(0, nGenTaus):
            genVisTauP4 = genTaus[i].getvisMomentum()
            genTauId = genTaus[i].getID()

            #   # # P4 Tau filters
            # if genVisTauP4.P()<5: continue
            # if abs(math.cos(genVisTauP4.Theta())>0.9): continue

            # pick only a decay mode in particular if you want
            if selectDecay != -777 and selectDecay != genTauId:
                continue
            findMatch, nTausType = tauReco.MatchRecoGenTau(
                genTaus[i],
                recoTaus,
                nTausType,
                maxDRMatch=experiment_parameters["MatchedGenMinDR"],
                selectDecay=selectDecay,
            )
            if not matched_cm:
                true_predicted_label["GenID"].append(str(eventid) + str(i))
                true_predicted_label["True"].append(genTauId)

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

            recoTauId = recoTaus[findMatch].getID()
            recoDM = recoTauId
            n_pi0s = 0
            nPhotons = 0
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

    logger_process.info("Found %d events", countEvents)

    decaystr = "decayAll" if selectDecay == -777 else "decay{}".format(selectDecay)
    exp_str = config["experiment"] + "_" + str(exp_value)
    true_predicted_label_output_file = (
        outputpath + f"true_predicted_label_{decaystr}_{exp_str}.csv"
    )
    true_predicted_label_df = pd.DataFrame(true_predicted_label)
    true_predicted_label_df.to_csv(true_predicted_label_output_file, index=False)

    # Check if config["output"]["outputlabels"] is a list
    if type(config["output"]["outputlabels"]) is not list:
        if config["output"]["outputlabels"] is None:
            config["output"]["outputlabels"] = []
        else:
            config["output"]["outputlabels"] = [config["output"]["outputlabels"]]
    if true_predicted_label_output_file not in config["output"]["outputlabels"]:
        config["output"]["outputlabels"].append(true_predicted_label_output_file)

output_config_file = outputpath+"config.yaml"
# print(config)
with open(output_config_file, "w") as file:
    yaml.dump(config, file)
    logger_config.info("Configuration saved to '%s'", output_config_file)
    # print(f"Saved configuration parameters to '{output_config_file}'.")

logger_process.info("Experiment finished!")
