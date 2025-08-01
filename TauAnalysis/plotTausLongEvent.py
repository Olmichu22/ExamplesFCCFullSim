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
import json
from modules import ParticleObjects
from modules.ParticleObjects import RecoParticle

from modules import pi0Reco
from modules import tauReco
from modules import electronReco
from modules import muonReco
from modules import myutils

import logging


def extend_parser(parser):
    parser.add_argument("--nfile", type=int, default=1, help="Id of simulation file")
    parser.add_argument("--eventid", type=int, default=0, help="Event ID to analyze")


outputbasepath = "Event_info/TauReco/"
default_config = "config/default/taurecolong.yaml"

general_configs = myutils.setup_analysis_config(
    default_config,
    outputbasepath,
    parser_hook=extend_parser,
)

args = general_configs["args"]
config = general_configs["config"]
loggers = general_configs["loggers"]
logger_config = loggers["config"]
logger_io = loggers["io"]
logger_process = loggers["processing"]
logger_pi0mass = loggers["pi0mass"]
selectDecay = general_configs["decay"]

def _first(val):
    return val[0] if isinstance(val, list) else val

dRMax = _first(config["cuts"]["dRMax"])
minPTauPhoton = _first(config["cuts"]["TauPhotonPCut"])
minPTauPion = _first(config["cuts"]["TauPionPCut"])
PNeutron = _first(config["cuts"]["NeutronCut"])
dRMatch = _first(config["cuts"]["MatchedGenMinDR"])
generalPCut = _first(config["cuts"]["generalPCut"])

outfile = config["general"]["outfile"]

# Output Configuration
outputbasepath = "Event_info/TauReco/"


cut_string = f"_{dRMax}_tph{minPTauPhoton}_tpi{minPTauPion}_n{PNeutron}_g{generalPCut}"
decayString = f"decay{selectDecay}" + cut_string
if selectDecay == -777:
    decayString = "decayAll" + cut_string
fileOutName = outfile + "event_" + str(args.eventid)+"_" + decayString + ".root"


# Different output paths depending on the GATr result and test PFO
# if args.gatr_result is not None and args.test_pfo:
    # outputpath = outputbasepath + "PFO_"+ outfile + cut_string[1:] + "/"
# elif args.gatr_result is None and args.test_pfo:
    # logger_config.error("Cannot use --test-pfo without --gatr-result.")
    # raise ValueError("Cannot use --test-pfo without --gatr-result.")
# else:
    # outputpath = outputbasepath + "File_" + args.nfile + outfile + cut_string[1:] + "/"
outputpath = outputbasepath + "File_" + str(args.nfile)+"_" + outfile + cut_string[1:] + "/"

# if args.gatr_result is not None:
#     outputpath = "GATr_" + outputpath

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

# ------------------------------------------------------------------------
# Logging configuration
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
    stream_handler.setLevel(logging.DEBUG)  # Terminal muestra DEBUG o superior
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

# ------------------------------------------------------------------------
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



# ------------------------------------------------------------------------
# GATr reading config (if provided)

gatr_results_path = args.gatr_result


# Simulation files
path = "/pnfs/ciemat.es/data/cms/store/user/cepeda/FCC/FullSim/"
file = "out_reco_edm4hep_edm4hep"
nfile = args.nfile
file = file + "_{}".format(nfile) + ".root"
dir_path = path + "/" + sample

file_path = dir_path + "/" + file



filenames = []
my_file = Path(file_path)
if my_file.is_file():
    root_file = myutils.open_root_file(file_path)
    if not root_file or root_file.IsZombie():
        logger_io.error("File %s is a zombie or could not be opened.", file_path)
        raise FileNotFoundError(f"File {file_path} not found or is a zombie.")
    filenames.append(file_path)
reader = root_io.Reader(filenames)
logger_io.info("Read %d files", len(filenames))
logger_io.info("First %s files.", filenames) 

result_labels = {}
result_labels["tau1"] = []
result_labels["tau2"] = []
result_labels["id-tau1"] = []
result_labels["id-tau2"] = []

# unmatched_reco_pions_match_list = []
# unmatched_reco_pions_P_per_miss = {"Non_matched":hUnmatchedRecoPionsP}

anal_event_id = args.eventid - 1

def get_particle_info_dict(particle, gen = False, daughter=False):
    """Extracts relevant information from a particle object."""
    if not daughter:
        results = {
            "PDGID": particle.getPDG(),
            "Decay Type": particle.getID(),
            "Px": particle.getMomentum().X(),
            "Py": particle.getMomentum().Y(),
            "Pz": particle.getMomentum().Z(),
            "Mass": particle.getMass(),
            "P": particle.getMomentum().P(),
            "Theta": particle.getMomentum().Theta(),
            "Phi": particle.getMomentum().Phi(),
        }
        if gen:
            results["visPx"] = particle.getvisMomentum().X()
            results["visPy"] = particle.getvisMomentum().Y()
            results["visPz"] = particle.getvisMomentum().Z()
            results["visMass"] = particle.getVisMass()
            results["visP"] = particle.getvisMomentum().P()
            results["visTheta"] = particle.getvisMomentum().Theta()
            results["visPhi"] = particle.getvisMomentum().Phi()
    else:
        results = {
            "PDGID": particle.getPDG(),
            "Px": particle.getMomentum().x,
            "Py": particle.getMomentum().y,
            "Pz": particle.getMomentum().z,
            "Mass": particle.getMass(),
        }
        P4Momentum = ROOT.TLorentzVector(
            particle.getMomentum().x,
            particle.getMomentum().y,
            particle.getMomentum().z,
            particle.getMass(),
        )
        results["P"] = P4Momentum.P()
        results["Theta"] = P4Momentum.Theta()
        results["Phi"] = P4Momentum.Phi()
    return results
    
# collections to use
genparts = "MCParticles"
pfobjects = "PandoraPFOs"

for eventid, event in enumerate(reader.get("events")):
    # if gatr_results_path is not None and eventid > len(gatr_results) - 1:
    #     logger_process.info("Reached the end of GATr results, stopping processing.")
    #     break
    if eventid != anal_event_id:
        continue
    
    logger_process.debug("Processing event %d", eventid)

    mc_particles = event.get(genparts)
    pfos = event.get(pfobjects)

    genTaus = tauReco.findAllGenTaus(mc_particles)
    nGenTaus = len(genTaus)

    logger_process.debug(
        "Found %d gen taus. Details:\n%s",
        nGenTaus,
        "\n".join("GenTau %d: %s" % (i, tau) for i, tau in genTaus.items()),
    )

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
    # nRecoTaus = len(recoTaus)
    logger_process.debug(
        "Found %d reco taus, electrons and muons. Details:\n%s",
        pidx,
        "\n".join("RecoParticle %d: %s" % (i, tau) for i, tau in recoTaus.items()),
    )
    results = {"Gen Info": [], "Reco Info": []}
    genTausID = {}
    genDaughtersPDG = {}
    recoTausID = {}
    recoDaughtersPDG = {}
    for i, gentau in genTaus.items():
        gen_info = get_particle_info_dict(gentau, gen=True)
        logger_process.debug(f"Gen Tau {i} info: {gen_info}")
        results["Gen Info"].append(gen_info)
        genTausID[gentau.getID()] = genTausID.get(gentau.getID(), 0) + 1

        results["Gen Info"][-1]["Daughters"] = []
        # print(results["Gen Info"][-1])
        for i, daughter in gentau.getDaughters().items():
            results["Gen Info"][-1]["Daughters"].append(get_particle_info_dict(daughter, gen=True, daughter=True))
            genDaughtersPDG[abs(daughter.getPDG())] = genDaughtersPDG.get(abs(daughter.getPDG()), 0) + 1
    for i, recotau in recoTaus.items():
        reco_info = get_particle_info_dict(recotau)
        logger_process.debug(f"Reco Tau {i} info: {reco_info}")
        results["Reco Info"].append(reco_info)
        recoTausID[recotau.getID()] = recoTausID.get(recotau.getID(), 0) + 1
        
        results["Reco Info"][-1]["Daughters"] = []
        for i, daughter in recotau.getDaughters().items():
            results["Reco Info"][-1]["Daughters"].append(get_particle_info_dict(daughter, daughter=True))
            recoDaughtersPDG[abs(daughter.getPDG())] = recoDaughtersPDG.get(abs(daughter.getPDG()), 0) + 1
    
    results["Summary"] = {
        "Gen Taus ID": genTausID,
        "Gen Daughters PDG": genDaughtersPDG,
        "Reco Taus ID": recoTausID,
        "Reco Daughters PDG": recoDaughtersPDG,
        "nRecoHadronTaus": nRecoTaus,
        "nRecoElectrons": nRecoElectrons,
        "nRecoMuons": nRecoMuons,
        "nGenTaus": nGenTaus,
    }
    


# logger_process.info("Found %d events", countEvents)

# decaystr = "decayAll" if selectDecay == -777 else "decay{}".format(selectDecay)
# true_predicted_label_output_file = outputpath + f"true_predicted_label_{decaystr}.csv"
# true_predicted_label_df = pd.DataFrame(true_predicted_label)
# true_predicted_label_df.to_csv(true_predicted_label_output_file, index=False)

# result_labels = pd.DataFrame(result_labels)
# result_labels.to_csv(outputpath + "result_labels_pfo.csv", index=False)

# unmatched_reco_pions_match_list_df = pd.DataFrame({"PDGID": unmatched_reco_pions_match_list})
# unmatched_reco_pions_match_count = unmatched_reco_pions_match_list_df["PDGID"].value_counts()
# # Guardamos
# unmatched_reco_pions_match_count.to_csv(outputpath + "unmatched_reco_pions_match_count.csv", index=True) 

# # Check if config["output"]["outputlabels"] is a list
# if type(config["output"]["outputlabels"]) is not list:
#     if config["output"]["outputlabels"] is None:
#         config["output"]["outputlabels"] = []
#     else:
#         config["output"]["outputlabels"] = [config["output"]["outputlabels"]]
# if true_predicted_label_output_file not in config["output"]["outputlabels"]:
#     config["output"]["outputlabels"].append(true_predicted_label_output_file)

event_info_path = outputpath + "event_info_" +"file_" + str(args.nfile)+"_event_" +str(args.eventid) +".json"

with open(event_info_path, "w") as file:
    json.dump(results, file, indent=4)
    logger_io.info("Event information saved to %s", event_info_path)

output_config_file = outputpath + "config.yaml"
with open(output_config_file, "w") as file:
    yaml.dump(config, file)
    logger_io.info("Configuration file saved to %s", output_config_file)


# logger_io.info("Output file %s", outputpath + fileOutName)
logger_io.info("End of job")
# outfile.Close()
