import sys
import math
import ROOT
import yaml
import os
import argparse
from array import array
from podio import root_io
import edm4hep
import logging
import pandas as pd
import pickle
import numpy as np
import pprint
logger_io = logging.getLogger('io')
def associate_reco_with_gen_taus(gen_taus, reco_tau):
    """Asocia cada hemisferio con el tau correspondiente usando la dirección del tau."""
    
    # Obtener dirección de cada tau
    tau_directions = []
    for key, tau in gen_taus.items():
        px = tau.getMomentum().X()
        py = tau.getMomentum().Y()
        pz = tau.getMomentum().Z()
        tau_directions.append((px, py, pz))
    
    reco_tau_direction = [reco_tau.getMomentum().X(),
                          reco_tau.getMomentum().Y(),
                          reco_tau.getMomentum().Z()]
    
    # Calcular cosenos de ángulos entre direcciones
    cos_r_tau1 = np.dot(reco_tau_direction, tau_directions[0]) / (np.linalg.norm(reco_tau_direction) * np.linalg.norm(tau_directions[0]))
    cos_r_tau2 = np.dot(reco_tau_direction, tau_directions[1]) / (np.linalg.norm(reco_tau_direction) * np.linalg.norm(tau_directions[1]))
    
    # El hemisferio 1 corresponde al tau 1 si el coseno es mayor
    if cos_r_tau1 > cos_r_tau2:
        return list(gen_taus.keys())[0], cos_r_tau1
    else:
        return list(gen_taus.keys())[1], cos_r_tau2
    
# I'm sure this exists already 
def dRAngle(p1,p2):
    """ Calculate the angle between two particles in the eta-phi plane
    Args:
    p1 (TLorentzVector): 4-momentum vector of the first particle
    p2 (TLorentzVector): 4-momentum vector of the second particle
    Returns:
    float: angle between the two particles in the theta-phi plane
    """
    dphi=p1.Phi()-p2.Phi()
    if (dphi>math.pi) : dphi=2*math.pi-dphi
    if (dphi<-math.pi) : dphi=2*math.pi+dphi
    dtheta=p1.Theta()-p2.Theta()
    dR=math.sqrt(dtheta*dtheta+dphi*dphi)
    return dR

# trick to prevent broken files (should not be a problem at CIEMAT)
def open_root_file(file_path):
    try:
        # Suppress ROOT's default error messages to the terminal
        ROOT.gErrorIgnoreLevel = ROOT.kError

        # Attempt to open the ROOT file in "READ" mode without auto-recovery
        root_file = ROOT.TFile.Open(file_path, "READ")

        # Check if the file is a zombie
        if not root_file or root_file.IsZombie():
            logger_io.error(f"Error: '{file_path}' is a zombie or could not be opened.")
            raise IOError(f"Error: '{file_path}' is a zombie or could not be opened.")
        
        # Check if file is recoverable (potentially corrupted)
        if root_file.TestBit(ROOT.TFile.kRecovered):
            logger_io.error(f"Warning: '{file_path}' is corrupted and has been recovered.")
            raise IOError(f"Error: '{file_path}' is corrupted and has been recovered.")
        
        #print(f"'{file_path}' opened successfully.")
        logger_io.debug("File '%s' opened successfully.", file_path)
        return root_file

    except Exception as e:
        logger_io.error("Error opening file '%s': %s", file_path, e)
        # print(f"Error: {e}")
        return None

# Fuction to sort by tau P
def sort_by_P(Tau):
    tau_with_P = []

    for i in range(0,len(Tau)):
        tau_with_P.append((Tau[i], Tau[i].getMomentum().P()))
    
    # Sort the list based on the P() value in descending order
    sorted_tau_with_P = sorted(tau_with_P, key=lambda x: x[1], reverse=True)
    
    # Extract only the sorted Tau[i] objects from the tuples
    sortedTau = [tau for tau, _ in sorted_tau_with_P]
   
    return sortedTau

def load_yaml_config(config_file, default_config):
    """Load the YAML configuration file if it exists.
    Args:
    args (argparse.Namespace): command line arguments
    config_file (str): path to the YAML configuration file
    Returns:
    dict: configuration parameters
    """
    if config_file is not None and os.path.exists(config_file):
        with open(config_file, "r") as file:
            config = yaml.safe_load(file)
            # print(f"Loaded configuration parameters from '{config_file}'.")
    elif default_config:
        if not os.path.exists(default_config):
            raise FileNotFoundError(f"Error: '{default_config}' does not exist. A valid default configuration file is required.")
        with open(default_config, "r") as file:
            config = yaml.safe_load(file)
            # print(f"Loaded default configuration parameters from '{default_config}'.")
    else:
        raise FileNotFoundError(f"Error: A valid default configuration file is required.")
    return config

def load_yaml_config(config_file, default_config):
    """Load the YAML configuration file if it exists.
    Args:
    args (argparse.Namespace): command line arguments
    config_file (str): path to the YAML configuration file
    Returns:
    dict: configuration parameters
    """
    if config_file is not None and os.path.exists(config_file):
        with open(config_file, "r") as file:
            config = yaml.safe_load(file)
            # print(f"Loaded configuration parameters from '{config_file}'.")
    elif default_config:
        if not os.path.exists(default_config):
            raise FileNotFoundError(f"Error: '{default_config}' does not exist. A valid default configuration file is required.")
        with open(default_config, "r") as file:
            config = yaml.safe_load(file)
            # print(f"Loaded default configuration parameters from '{default_config}'.")
    else:
        raise FileNotFoundError(f"Error: A valid default configuration file is required.")
    return config




def setup_analysis_config(
    default_config: str = "config/default/taurecolong.yaml",
    output_base: str = "Results/TauReco/"
):
    """
    Encapsula la configuración de argumentos, cargas de configuración,
    aplicación de cortes, configuración de rutas de salida y logging.

    Devuelve un diccionario con keys:
      - args: Namespace de argparse
      - config: diccionario de configuración actualizado
      - outputpath: ruta de salida creada
      - fileOutName: nombre de archivo de salida
      - loggers: diccionario con loggers (config, io, processing, pi0mass)
    """
    # Argument parser setup
    parser = argparse.ArgumentParser(
        description="Configure the analysis",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    
    
    
    parser.add_argument("-f", "--sample")
    parser.add_argument("-o", "--outfile")
    parser.add_argument("-d", "--decay", type=int)
    parser.add_argument("-p", "--TauPhotonPCut", type=float)
    parser.add_argument("-i", "--TauPionPCut", type=float)
    parser.add_argument("-t","--tauCut",default=2,type=float)
    parser.add_argument("-R", "--dRMax", type=float)
    parser.add_argument("-n", "--NeutronCut", type=float)
    parser.add_argument("-g", "--generalPCut", type=float)
    parser.add_argument("-r", "--MatchedGenMinDR", type=float)
    parser.add_argument(
        "-m", "--matchedCM",
        default="True",
        type=str,
        help="Use only matched taus to compute confusion matrix.",
    )
    parser.add_argument(
        "--test",
        type=str,
        help="Run in test mode with limited number of files",
    )
    parser.add_argument(
        "-c", "--config", type=str, help="Configuration file"
    )
    parser.add_argument(
        "-v", "--verbose",
        action="count",
        default=0,
        help="Increase verbosity level: -v for INFO, -vv for DEBUG",
    )
    parser.add_argument(
        "--gatr-result",
        type=str,
        help="Path to GATR result for the analysis.",
    )
    parser.add_argument(
        "--test-pfo",
        action="store_true",
        help="Use this flag to test the PFOs in same files as GATr.",
    )

    args = parser.parse_args()

    # Load config
    config = load_yaml_config(args.config, default_config)

    # Cut Configuration
    cuts = config.get("cuts", {})
    for key in ["tauCut", "dRMax", "TauPhotonPCut", "TauPionPCut", "NeutronCut", "MatchedGenMinDR", "generalPCut"]:
        val = getattr(args, key) if getattr(args, key) is not None else cuts.get(key)
        cuts[key] = val
    config["cuts"] = cuts

    # Decay selection
    decay_list = config.setdefault("general", {}).setdefault("decay", [])
    select_decay = args.decay if args.decay is not None else decay_list[0]
    if args.decay is not None and args.decay not in decay_list:
        decay_list.append(args.decay)
    config["general"]["decay"] = decay_list

    # Output filename
    outfile = args.outfile or config["general"].get("outfile")
    config["general"]["outfile"] = outfile

    # Build strings
    dr = cuts["dRMax"]
    tph = cuts["TauPhotonPCut"]
    tpi = cuts["TauPionPCut"]
    npe = cuts["NeutronCut"]
    gpc = cuts["generalPCut"]
    mdr = cuts["MatchedGenMinDR"]

    suffix = f"_{dr}_tph{tph}_tpi{tpi}_n{npe}_g{gpc}"
    decay_str = f"decay{select_decay}" + suffix
    if select_decay == -777:
        decay_str = "decayAll" + suffix
    file_out = f"{outfile}{decay_str}.root"

    # Output path logic
    base = output_base + outfile + suffix[1:] + "/"
    if args.gatr_result and args.test_pfo:
        path = output_base + "PFO_" + outfile + suffix[1:] + "/"
    elif args.test_pfo:
        raise ValueError("Cannot use --test-pfo without --gatr-result.")
    else:
        path = base
    if args.gatr_result:
        path = "GATr_" + path
    os.makedirs(path, exist_ok=True)

    config.setdefault("output", {}).setdefault("outputfile", [])
    if not config["output"].get("outputfile") is None:
        if file_out not in config["output"]["outputfile"]:
            config["output"]["outputfile"].append(file_out)
    else:
        config["output"]["outputfile"] = [file_out]
    config["output"]["outputpath"] = path

    # Logging
    lvl = logging.WARNING if args.verbose == 0 else logging.INFO if args.verbose == 1 else logging.DEBUG
    handlers = []
    if args.verbose < 2:
        handlers = [logging.StreamHandler(sys.stdout), logging.FileHandler(os.path.join(path, "app.log"), mode="w")]
    elif args.verbose == 2:
        sh = logging.StreamHandler(sys.stdout); sh.setLevel(logging.DEBUG)
        fh = logging.FileHandler(os.path.join(path, "app.log"), mode="w"); fh.setLevel(logging.DEBUG)
        handlers = [sh, fh]
    else:
        handlers = [logging.FileHandler(os.path.join(path, "app.log"), mode="w")]

    logging.basicConfig(
        level=lvl,
        format="%(asctime)s, %(levelname)s, [%(name)s] - %(message)s",
        handlers=handlers
    )

    loggers = {
        "config": logging.getLogger("config"),
        "io": logging.getLogger("io"),
        "processing": logging.getLogger("processing"),
        "pi0mass": logging.getLogger("pi0mass")
    }

    # General args to config
    for key in ["sample", "matchedCM", "test"]:
        config["general"][key] = getattr(args, key) if getattr(args, key) is not None else config["general"].get(key)

    # Convert flags
    matched_cm = True if config["general"]["matchedCM"] == "True" else False
    test_mode = True if config["general"]["test"] == "True" else False

    loggers["config"].info("Configuration loaded!")
    loggers["config"].info("Configuration:\n%s", pprint.pformat(config, indent=4))

    return {
        "args": args,
        "config": config,
        "outputpath": path,
        "fileOutName": file_out,
        "loggers": loggers,
        "decay": select_decay,
        "flags": {
            "matched_cm": matched_cm,
            "test": test_mode
        }
    }
