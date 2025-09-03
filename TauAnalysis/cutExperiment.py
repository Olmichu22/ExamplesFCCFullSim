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
import pickle
from modules import tauReco 
from modules import myutils
from modules import ParticleObjects, electronReco, muonReco, myutils, pi0Reco, tauReco
def extend_parser(parser):
    parser.add_argument(
        "-V",
        "--value",
        type=str,
        help="Value to modify. It has to be one of the available cuts: NeutronCut, TauPhotonPCut, TauPionPCut, dRMax, MatchedGenMinDR, generalPCut",
    )
    parser.add_argument(
        "-l", "--range", nargs="+", type=float,
        help="Range of the value to modify: min, max, step"
    )


outputbasepath = "Results/Experiments/"
default_config = "config/default/cutexperiment.yaml"

general_configs = myutils.setup_analysis_config(
    default_config,
    outputbasepath,
    parser_hook=extend_parser,
    exp = True
)

args = general_configs["args"]
config = general_configs["config"]
# loggers = general_configs["loggers"]
# logger_config = loggers["config"]
# logger_io = loggers["io"]
# logger_process = loggers["processing"]
# logger_config.info("Logging handler initialized.")

values = ["NeutronCut", "TauPhotonPCut", "TauPionPCut", "dRMax", "MatchedGenMinDR", "generalPCut"]

# Set User Config (if any)
config["experiment"] = args.value if args.value is not None else config.get("experiment")

if config["experiment"] not in values:
    raise Exception(
        f"Value '{config['experiment']}' not valid. Choose between {values}"
    )


for key in list(config["cuts"].keys()):
    if key == config["experiment"]:
        if args.range is not None:
            if len(args.range) != 3:
                raise Exception(
                    f"Invalid range format. Expected format: min, max, step. Got {len(args.range)} values."
                )
            config["cuts"][key] = np.arange(args.range[0], args.range[1], args.range[2]).tolist()
        else:
            val = config["cuts"][key]
            if not isinstance(val, list):
                config["cuts"][key] = [val]
    else:
        val = config["cuts"][key]
        if isinstance(val, list):
            val = val[0]
        config["cuts"][key] = val


def _first(val):
    return val[0] if isinstance(val, list) else val

dRMax = _first(config["cuts"]["dRMax"])
minPTauPhoton = _first(config["cuts"]["TauPhotonPCut"])
minPTauPion = _first(config["cuts"]["TauPionPCut"])
PNeutron = _first(config["cuts"]["NeutronCut"])
dRMatch = _first(config["cuts"]["MatchedGenMinDR"])
generalPCut = _first(config["cuts"]["generalPCut"])


exp_str = ""
for key, value in config["cuts"].items():
    if key == config["experiment"]:
        exp_str += f"{key}Exp_"
    else:
        exp_str += f"{key}{value}_"
cut_string = f"_{exp_str[:-1]}"

selectDecay = general_configs["decay"]
decayString = f"decay{selectDecay}" + cut_string
if selectDecay == -777:
    decayString = "decayAll" + cut_string

config["general"]["outfile"] = (
    args.outfile if args.outfile != None else config["general"]["outfile"]
)
outfile = config["general"]["outfile"]
if args.test_pfo:
    outfile = "PFO_" + outfile   

fileOutName = outfile + decayString + ".root"

outputpath = outputbasepath + outfile + cut_string + "/"
if args.gatr_result is not None:
    outputpath = "GATr_" + outputpath
    
if not os.path.exists(outputpath):
    os.makedirs(outputpath)

# Once set the output path, we can set the logger
lvl = logging.WARNING if args.verbose == 0 else logging.INFO if args.verbose == 1 else logging.DEBUG
handlers = []
if args.verbose < 2:
    handlers = [logging.StreamHandler(sys.stdout), logging.FileHandler(os.path.join(outputpath, "exp.log"), mode="w")]
elif args.verbose == 2:
    sh = logging.StreamHandler(sys.stdout); sh.setLevel(logging.DEBUG)
    fh = logging.FileHandler(os.path.join(outputpath, "exp.log"), mode="w"); fh.setLevel(logging.DEBUG)
    handlers = [sh, fh]
else:
    handlers = [logging.FileHandler(os.path.join(outputpath, "exp.log"), mode="w")]

logging.basicConfig(
    level=lvl,
    format="%(asctime)s, %(levelname)s, [%(name)s] - %(message)s",
    handlers=handlers
)

logger_config = logging.getLogger("config")
logger_io = logging.getLogger("io")
logger_process = logging.getLogger("processing")
logger_config.info("Logging handler initialized.")
loggers = {
    "config": logger_config,
    "io": logger_io,
    "processing": logger_process
}

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

gatr_results_path = args.gatr_result

filenames, mlpf_results = myutils.get_root_trees_path(sample, gatr_results_path, loggers, test_arg)

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
        if gatr_results_path is not None and not general_configs["args"].test_pfo:
            logger_io.debug("Using GATr results for tau reconstruction")
            particles = mlpf_results.get(eventid, {})
            recoTau = tauReco.findAllTaus(particles,
                                    experiment_parameters["dRMax"],
                                    experiment_parameters["TauPhotonPCut"],
                                    experiment_parameters["TauPionPCut"],
                                    experiment_parameters["NeutronCut"],
                                    experiment_parameters["generalPCut"],
                                    charge_condition=False)
            recoElectrons = electronReco.findAllElectrons(particles, experiment_parameters["generalPCut"])
            recoMuons = muonReco.findAllMuons(particles, experiment_parameters["generalPCut"])
        else:
            logger_io.debug("Using Pandora results for tau reconstruction")
            recoTau = tauReco.findAllTaus(pfos,
                                experiment_parameters["dRMax"],
                                experiment_parameters["TauPhotonPCut"],
                                experiment_parameters["TauPionPCut"],
                                experiment_parameters["NeutronCut"],
                                experiment_parameters["generalPCut"])
            recoElectrons = electronReco.findAllElectrons(pfos,
                                                          experiment_parameters["generalPCut"])
            recoMuons = muonReco.findAllMuons(pfos,
                                              experiment_parameters["generalPCut"])
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
            # if not matched_cm:
            true_predicted_label["GenID"].append(str(eventid) + str(i))
            true_predicted_label["True"].append(genTauId)

            if findMatch == -1:
                logger_process.debug("No match found for gen tau %s", genTaus[i])
                # if not matched_cm:
                #     # true_predicted_label["Predicted"].append(-1)
                true_predicted_label["Predicted"].append(-2)
                true_predicted_label["PhotonPredicted"].append(-2)
                continue

            logger_process.debug("Found matched tau. Details:\n%s", recoTaus[findMatch])
            # if matched_cm:
            #     true_predicted_label["GenID"].append(str(eventid) + str(i))
            #     true_predicted_label["True"].append(genTauId)

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
