import sys, os, math 
from array import array
import ROOT
from ROOT import TFile, TTree, TH1F, TH2F
import numpy as np
from podio import root_io
import edm4hep
from pathlib import Path
import pandas as pd
from modules import tauReco
from modules import ZReco
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

decayString="Event_dist_decay"+str(selectDecay)+"_"+str(dRMax)+"_"+str(args.photonCut)+"_"+str(PNeutron)
if selectDecay==-777:
    decayString="Event_dist_decayall_"+str(dRMax)+"_"+str(args.photonCut)+"_"+str(PNeutron)
fileOutName=args.outfile+decayString+".root"

print ("=====================================")

# get all the files 
path="/pnfs/ciemat.es/data/cms/store/user/cepeda/FCC/FullSim/"
file="out_reco_edm4hep_edm4hep"
filenames=[]
dir_path=path+"/"+sample
names = ROOT.std.vector('string')()
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
pfobjects ="PandoraPFOs"
#pfobjects ="TightSelectedPandoraPFOs"

hRecoEventDist = TH1F("histRecoCardEventDist", "Number of Leptons per Event", 10, 0, 10) 

print ("-------------------------------------")
print ("Start processing!")

countEvents = 0

for event in reader.get("events"):

  if countEvents%500==0:
    print ("... %d" %countEvents)
  countEvents+=1

  # get the constituents
  mc_particles = event.get( genparts )
  pfos = event.get(pfobjects)

  # get the number of leptons in the event
  recoTaus= tauReco.findAllTaus(pfos,dRMax, minP,PNeutron)
  nTaus=len(recoTaus)

  


  for j in range(0,nTaus):
    recoTauP4=recoTaus[j][0]
    recoTauId=recoTaus[j][1]
    recoTauQ=recoTaus[j][2]
      #recoTauDR=recoTaus[j][3]
      #recoTauNConsts=recoTaus[j][4]
      #recoTauConsts=recoTaus[j][5]

      # to make the code more economic we are checking gen and reco in parallel, but 
      # there is a difference in the DM labelling:
      # at reco level we count photons and at gen level pi0s: difference in the
      # decay mode (1 gen can be 1 or 2 reco, etc )
    recoDM=recoTauId
    if recoTauId==2:
      recoDM=1
    elif recoTauId>=3 and recoTauId<10:
      recoDM=3
    elif (recoTauId>=11 and recoTauId<15):
      recoDM=11

    if selectDecay!=-777 and selectDecay!=recoDM:
      continue


print ("-------------------------------------")
print ("Processed %d events" %countEvents)
print ("Plots saved in %s" %fileOutName)
print ("=====================================")