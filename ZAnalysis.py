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

args = parser.parse_args()
config = vars(args)
print(config)

dRMax=args.dRMax
minP=args.photonCut
selectDecay=args.decay
fileOutName=args.outfile
PNeutron=args.neutronCut
selectDecay=args.decay
sample=args.sample
test= True if args.test=="True" else False

decayString="Ztau_decay"+str(selectDecay)+"_"+str(dRMax)+"_"+str(args.photonCut)+"_"+str(PNeutron)
if selectDecay==-777:
    decayString="Ztau_decayAll_"+str(dRMax)+"_"+str(args.photonCut)+"_"+str(PNeutron)
fileOutName=args.outfile+decayString+".root"

# get all the files 
path="/pnfs/ciemat.es/data/cms/store/user/cepeda/FCC/FullSim/"
file="out_reco_edm4hep_edm4hep"
filenames=[]
dir_path=path+"/"+sample
nfiles=len(os.listdir(dir_path))

nfiles=1000 #Maxium files to read
if test==True:
   nfiles=5
   
print ("Reading files from %s" %dir_path)
for i in range(1,nfiles+1):
    filename=dir_path+"/"+file+"_{}.root".format(i)
    print (filename)
    my_file = Path(filename)
    if my_file.is_file():
        root_file = myutils.open_root_file(filename)
        if not root_file or root_file.IsZombie():
            continue
        filenames.append(filename)

print ("Read %d files" %len(filenames))
reader = root_io.Reader(filenames)

# collections to use 
genparts = "MCParticles"
pfobjects="PandoraPFOs"

hGenZP = TH1F("histoGenZP", "Gen Z momentum", 50, 0, 100)
hGenVisZP = TH1F("histoGenVisZP", "Gen Z visible momentum", 50, 0, 100)
hGenVisZMass