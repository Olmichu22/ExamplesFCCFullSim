import sys, os, math 
from array import array
import ROOT
from ROOT import TFile, TTree, TH1F, TH2F
import numpy as np
from podio import root_io
import edm4hep
from pathlib import Path

from modules import tauReco
from modules import myutils 

import argparse
parser = argparse.ArgumentParser(description="Configure the analysis",
                         formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("-f", "--sample", default="ZTauTau_SMPol_25Sept_MuonFix", help="Sample file name to process")
parser.add_argument("-o", "--outfile", default="firstTest_", help="Output file name prefix")
parser.add_argument("-d", "--decay", default=-777, type=int, help="Decay mode to select (-777 for all)")
parser.add_argument("-p", "--photonCut", default=0.1, type=float, help="Photon momentum cut value")
parser.add_argument("-R", "--dRMax", default=0.4, type=float, help="Maximum delta R value")
parser.add_argument("-n", "--neutronCut", default=1, type=float, help="Neutron momentum cut value")
parser.add_argument("-t", "--test", default="True", type=str, help="Run in test mode with limited number of files")

