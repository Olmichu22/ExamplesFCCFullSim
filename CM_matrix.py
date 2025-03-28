import sys, os, math 
from array import array
import ROOT
from ROOT import TFile, TTree, TH1F, TH2F
import numpy as np
from podio import root_io
import edm4hep
from pathlib import Path
from pprint import pprint
import yaml
import pandas as pd
import logging
from modules import tauRecoMaria

from modules import tauReco 
from modules import myutils 

import argparse
parser = argparse.ArgumentParser(description="Configure the analysis",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("-f","--sample")
parser.add_argument("-o","--outfile")
parser.add_argument("-d","--decay", type=int) # GEN 
parser.add_argument("-p","--TauPhotonPCut", type=float) # Creo que esto es un corte de momento en general
parser.add_argument("-i","--TauPionPCut", type=float)
parser.add_argument("-R","--dRMax", type=float)
parser.add_argument("-n","--NeutronCut", type=float)
parser.add_argument("-g","--generalPCut", type=float)
parser.add_argument("-r", "--MatchedGenMinDR", type=float)
parser.add_argument("-m", "--matchedCM", default="True", type=str, help="Use only matched taus to compute confusion matrix.")
parser.add_argument("-t", "--test", default="True", type=str, help="Run in test mode with limited number of files")
parser.add_argument("-c", "--config", default="config.yaml", type=str, help="Configuration file")

args = parser.parse_args()
default_config = "config/default/taurecolong.yaml"
config = myutils.load_yaml_config(args.config, default_config)

# Cut Configuration
config["cuts"]["dRMax"] = args.dRMax if args.dRMax != None else config["cuts"]["dRMax"]
config["cuts"]["TauPhotonPCut"] = args.TauPhotonPCut if args.TauPhotonPCut != None else config["cuts"]["TauPhotonPCut"]
config["cuts"]["TauPionPCut"] = args.TauPionPCut if args.TauPionPCut != None else config["cuts"]["TauPionPCut"]
config["cuts"]["NeutronCut"] = args.NeutronCut if args.NeutronCut != None else config["cuts"]["NeutronCut"]
config["cuts"]["MatchedGenMinDR"] = args.MatchedGenMinDR if args.MatchedGenMinDR != None else config["cuts"]["MatchedGenMinDR"]
config["cuts"]["generalPCut"] = args.generalPCut if args.generalPCut != None else config["cuts"]["generalPCut"]
dRMax=config["cuts"]["dRMax"]
minPTauPhoton=config["cuts"]["TauPhotonPCut"]
minPTauPion=config["cuts"]["TauPionPCut"]
PNeutron=config["cuts"]["NeutronCut"]
dRMatch=config["cuts"]["MatchedGenMinDR"]
generalPCut=config["cuts"]["generalPCut"]

# General Configuration

# We can use same config but different decay mode
# Priority is given to the decay mode in the command line
if args.decay not in config["general"]["decay"] and args.decay != None:
  config["general"]["decay"].append(args.decay)
  selectDecay = args.decay
else:
  selectDecay = args.decay if args.decay !=None else config["general"]["decay"][0]

config["general"]["sample"] = args.sample if args.sample!= None else config["general"]["sample"]
config["general"]["matchedCM"] = args.matchedCM if args.matchedCM != None else config["general"]["matchedCM"]
config["general"]["test"] = args.test if args.test != None else config["general"]["test"]
config["general"]["outfile"] = args.outfile if args.outfile!= None else config["general"]["outfile"]

sample = config["general"]["sample"]
matched_cm_arg = config["general"]["matchedCM"]
matched_cm = True if matched_cm_arg=="True" else False
test_arg = config["general"]["test"]
test = True if test_arg=="True" else False
outfile = config["general"]["outfile"]

# Output Configuration
outputbasepath = "Results/CM_TauReco/"


cut_string = f"_{dRMax}_tph{minPTauPhoton}_tpi{minPTauPion}_n{PNeutron}_g{generalPCut}"
decayString = f"decay{selectDecay}"+cut_string
if selectDecay==-777:
    decayString = "decayAll"+cut_string
fileOutName = outfile+decayString+".root"

outputpath = outputbasepath+outfile+cut_string[1:]+"/"

# Configuración básica del logger
logging.basicConfig(
    level=logging.DEBUG,  # El nivel que quieras mostrar (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    filename=outputpath+"debug.log",  # Descomenta esto si quieres que lo guarde en un fichero en lugar de pantalla
    filemode="w"
)

logger = logging.getLogger("TauRecoLogger")


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

print("Configuration:")
pprint(config, indent = 4)

print ("=====================================")



# get all the files 
path="/pnfs/ciemat.es/data/cms/store/user/cepeda/FCC/FullSim/"
file="out_reco_edm4hep_edm4hep"
filenames=[]
dir_path=path+"/"+sample
names = ROOT.std.vector('string')()
nfiles=len(os.listdir(dir_path))

nfiles=1000
if test==True:
   nfiles=20

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

print ("-------------------------------------")

hGenTauType = TH1F("hGenTauType", "GenTauType", 40, -20, 20)
hRecotauType = TH1F("hRecotauType", "RecotauType", 40, -20, 20)
hRecoGenTypeMatched = TH2F("hRecoGenTypeMatched","Matched Type",40, -20, 20, 32, -2, 20)
true_predicted_label = {"GenID":[],"True":[], "Predicted":[]}


# collections to use 
genparts = "MCParticles"
pfobjects="PandoraPFOs"
countEvents=0

for eventid, event in enumerate(reader.get("events")):
  if countEvents%1000==0:
    print (".... %d" %countEvents)
  countEvents+=1
    
  mc_particles = event.get(genparts)
  pfos = event.get(pfobjects)
  
  genTaus=tauReco.findAllGenTaus(mc_particles)
  nGenTaus=len(genTaus)
  logger.debug(f"Event {eventid}")
  logger.debug(f"Llamando a findAllTaus con dRMax={dRMax}, minPTauPhoton={minPTauPhoton}, minPTauPion={minPTauPion}")
  recoTaus = tauReco.findAllTaus(pfos,dRMax, minPTauPhoton, minPTauPion, PNeutron, generalPCut)
  # logger.debug(f"Llamando a findAllTaus María con dRMax={dRMax}, minP={generalPCut}, PNeutron={PNeutron}")
  # recoTaus2 = tauRecoMaria.findAllTaus(pfos,dRMax, generalPCut, PNeutron)
  nRecoTaus=len(recoTaus)
  nTausType = 0
  for i in range(0,nGenTaus):
    genVisTauP4=genTaus[i].getvisMomentum()
    genTauId=genTaus[i].getID()
    
      # # P4 Tau filters
    if genVisTauP4.P()<5: continue 
    if abs(math.cos(genVisTauP4.Theta())>0.9): continue
    
          # pick only a decay mode in particular if you want 
    if selectDecay!=-777 and selectDecay!=genTauId:
        continue
       
    hGenTauType.Fill(genTauId)

    findMatch, nTausType = tauReco.MatchRecoGenTau(genTaus[i], recoTaus, nTausType, maxDRMatch=dRMatch, selectDecay=selectDecay)
    
    if findMatch==-1:
      continue
    

    true_predicted_label["GenID"].append(str(eventid)+str(i))
    true_predicted_label["True"].append(genTauId) 
    
    recoTauId=recoTaus[findMatch].getID()
    hRecotauType.Fill(recoTauId)
    hRecoGenTypeMatched.Fill(genTauId, recoTauId)
    true_predicted_label["Predicted"].append(recoTauId)
    

print ("-------------------------------------")
print ("Processed %d events" %countEvents)

decaystr = "decayAll" if selectDecay==-777 else "decay{}".format(selectDecay)
true_predicted_label_output_file = outputpath+f"true_predicted_label_{decaystr}.csv"
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
with open(output_config_file, "w") as file:
    yaml.dump(config, file)
    print(f"Saved configuration parameters to '{output_config_file}'.")
print ("=====================================")
outfile=ROOT.TFile(outputpath+fileOutName,"RECREATE")
hGenTauType.Write()
hRecotauType.Write()
hRecoGenTypeMatched.Write()
outfile.Close()
print ("Output saved in %s" %outputpath+fileOutName)
print ("=====================================")