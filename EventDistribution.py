import sys, os, math 
from array import array
import ROOT
from ROOT import TFile, TTree, TH1F, TH2F
import numpy as np
from podio import root_io
import edm4hep
from pathlib import Path
import pandas as pd
from modules import tauReco, muonReco, electronReco
from modules import ZReco
from modules import myutils
import yaml
from pprint import pprint

import argparse
parser = argparse.ArgumentParser(description="Configure the analysis",
                         formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("-f", "--sample", help="Sample file name to process")
parser.add_argument("-o", "--outfile", help="Output file name prefix")
parser.add_argument("-d", "--decay", type=int, help="Decay mode to select (-777 for all)")
parser.add_argument("-p", "--TauPhotonPCut", type=float, help="Tau photon momentum cut value")
parser.add_argument("-i","--TauPionPCut", type=float, help="Tau pion momentum cut value")
parser.add_argument("-m", "--MuonPCut", type=float, help="Electron momentum cut value")
parser.add_argument("-e", "--ElectPCut", type=float, help="Muon momentum cut value")
parser.add_argument("-R", "--dRMax", type=float, help="Maximum delta R value")
parser.add_argument("-n", "--NeutronCut", type=float, help="Neutron momentum cut value")
parser.add_argument("-t", "--test", type=str, help="Run in test mode with limited number of files")
parser.add_argument("-c", "--config", default="config.yaml", type=str, help="Configuration file")

args = parser.parse_args()
default_config = "config/default/eventdist.yaml"
config = myutils.load_yaml_config(args.config, default_config)

dRMax=args.dRMax if args.dRMax != None else config["cuts"]["dRMax"]
minPTauPhoton=args.TauPhotonPCut if args.TauPhotonPCut != None else config["cuts"]["TauPhotonPCut"]
minPTauPion=args.TauPionPCut if args.TauPionPCut != None else config["cuts"]["TauPionPCut"]
minPMuon=args.MuonPCut if args.MuonPCut != None else config["cuts"]["MuonPCut"]
minPElectron=args.ElectPCut if args.ElectPCut != None else config["cuts"]["ElectPCut"]
PNeutron=args.NeutronCut if args.NeutronCut != None else config["cuts"]["NeutronCut"]

# We can use same config but different decay mode
# Priority is given to the decay mode in the command line
if args.decay not in config["general"]["decay"] and args.decay != None:
  config["general"]["decay"].append(args.decay)
  selectDecay = args.decay
else:
  selectDecay = args.decay if args.decay !=None else config["general"]["decay"][0]
  
sample=args.sample if args.sample!= None else config["general"]["sample"]
test_arg = args.test if args.test != None else config["general"]["test"]
test= True if test_arg=="True" else False
outfile=args.outfile if args.outfile!= None else config["general"]["outfile"]

outputbasepath = "Results/ZReco/"


cut_string = f"_{dRMax}_tph{minPTauPhoton}_tpi{minPTauPion}_m{minPMuon}_e{minPElectron}_n{PNeutron}"
decayString = f"decay{selectDecay}"+cut_string
if selectDecay==-777:
    decayString = "decayAll"+cut_string
fileOutName = outfile+decayString+".root"

outputpath = outputbasepath+outfile+cut_string[1:]+"/"

# Finish the configuration
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
dir_path=path+sample
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

hRecoEventDist = TH1F("histRecoNLeptonsEventDist", "Number of Leptons per Event", 10, 0, 10) 
hRecoEventMuonP = TH1F("histRecoEventMuonP", "Muon Momentum", 50, 0, 50)
hRecoEventElectronP = TH1F("histRecoEventElectronP", "Electron Momentum", 50, 0, 50)
hRecoEventTauP = TH1F("histRecoEventTauP", "Tau Momentum", 50, 0, 50)
hRecoPairCharge = TH1F("histRecoPairCharge", "Oposite Charge", 10, -5, 5)
hRecoEventCharge = TH1F("histRecoCharge", "Oposite Charge", 10, -5, 5)

hRecoEventMuonP_nLeptons = {}
hRecoEventElectronP_nLeptons = {}
hRecoEventTauP_nLeptons = {}

# pair_cases = {
#   "muonmuon": 0,
#   "muonelectron": 1,
#   "electronmuon": 1,
#   "muontau": 2,
#   "taumuon": 2,
#   "electrontau": 3,
#   "tauelectron": 3,
#   "tautau": 4,
# }
pair_cases_set = set()
pair_cases = {}
pair_cases_charge = {-2.0:0, 0:0, 2.0:0}
pair_cases_count = 0

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
  recoTaus= tauReco.findAllTaus(pfos, dRMax, minPTauPhoton, minPTauPion,PNeutron)
  recoMuons = muonReco.findAllMuons(pfos, minPMuon)
  recoElectrons = electronReco.findAllElectrons(pfos, minPElectron)
  nTaus=len(recoTaus)
  nMuons=len(recoMuons)
  nElectrons=len(recoElectrons)
  
  nLeptons=nMuons+nElectrons+nTaus
  
  
  case = str(nTaus)+"tau"+str(nMuons)+"muon"+str(nElectrons)+"electron"
  if case not in pair_cases_set:
    pair_cases_set.add(case)
    pair_cases[case] = 1
  else:
    pair_cases[case] += 1
  
  if nLeptons==0:
    continue
  
  if nLeptons not in hRecoEventMuonP_nLeptons.keys():
    hRecoEventElectronP_nLeptons[nLeptons] = TH1F(f"histRecoEventElectronP_{nLeptons}Leptons", f"Electron Momentum {nLeptons} Leptons", 50, 0, 50)
    hRecoEventMuonP_nLeptons[nLeptons] = TH1F(f"histRecoEventMuonP_{nLeptons}Leptons", f"Muon Momentum {nLeptons} Leptons", 50, 0, 50)
    hRecoEventTauP_nLeptons[nLeptons] = TH1F(f"histRecoEventTauP_{nLeptons}Leptons", f"Tau Momentum {nLeptons} Leptons", 50, 0, 50)
  
  recoLeptons = {}
  
  for i, muon in recoMuons.items():
    recoLeptons[f"muon{i}"] = muon
    hRecoEventMuonP.Fill(muon.getMomentum().P())
    hRecoEventMuonP_nLeptons[nLeptons].Fill(muon.getMomentum().P())
  for i, electron in recoElectrons.items():
    recoLeptons[f"electron{i}"] = electron
    hRecoEventElectronP.Fill(electron.getMomentum().P())
    hRecoEventElectronP_nLeptons[nLeptons].Fill(electron.getMomentum().P())  
    
  for i, tau in recoTaus.items():
    if tau.getID() == -13 or tau.getID() == -11:
      continue
    recoLeptons[f"tau{i}"] = tau
    hRecoEventTauP.Fill(tau.getMomentum().P())  
    hRecoEventTauP_nLeptons[nLeptons].Fill(tau.getMomentum().P())
  
  # fill histograms depending on the number of leptons
  tot_charge = 0
  pair_cases_count += 1
  for i, lepton in recoLeptons.items():
    tot_charge += lepton.getCharge()
  if nLeptons==2:
    hRecoPairCharge.Fill(tot_charge)
    if tot_charge == 0:
      pair_cases_charge[0] += 1
    else:
      pair_cases_charge[tot_charge] += 1
  hRecoEventCharge.Fill(tot_charge)
    
  hRecoEventDist.Fill(nLeptons)


# Create a DataFrame for the type of cases
max_case = len(pair_cases)
hRecoEventTypeDist = TH1F("histRecoCardEventTypeDist", "Type of Leptons per Event", max_case, 0, max_case)
event_type = pd.DataFrame(columns=["case", "number"])
event_type["case"] = pair_cases.keys()
event_type["number"] = pair_cases.values()

# Sort by number of cases
event_type = event_type.sort_values(by="number", ascending=False)
event_type["id"] = range(0, len(event_type))
for i, row in event_type.iterrows():
  hRecoEventTypeDist.Fill(row["id"], row["number"])


# Create a DataFrame for the charge of the pairs
pair_cases_charge_df = pd.DataFrame(columns=["charge", "number"])
pair_cases_charge_df["charge"] = pair_cases_charge.keys()
pair_cases_charge_df["number"] = pair_cases_charge.values()
pair_cases_charge_df.loc[-1] = ["total", pair_cases_charge_df["number"].sum()]
pair_cases_charge_df.index = pair_cases_charge_df.index + 1
pair_cases_charge_df.sort_index(inplace=True)

print(f"Total neutral pairs {pair_cases_charge[0]}")

print ("-------------------------------------")
print ("Processed %d events" %countEvents)
# Save Dicts and txt files


# Save keys (number of leptons) as txt file  
nleptons_keys = hRecoEventMuonP_nLeptons.keys()
output_key_file = outputpath+"nLeptons_keys.txt"
with open(output_key_file, "w") as f:
  for item in nleptons_keys:
    f.write("%s\n" % item)

output_event_type_file = outputpath+"/event_cases.csv"
event_type.to_csv(output_event_type_file, index=False)

output_pair_cases_charge_file = outputpath+"/pair_cases_charge.csv"
pair_cases_charge_df.to_csv(output_pair_cases_charge_file, index=False)

# Add the keys to the configuration
config["output"]["nLeptons_keys"] = output_key_file
config["output"]["event_cases"] = output_event_type_file
config["output"]["pair_cases_charge"] = output_pair_cases_charge_file

# Save the configuration
output_config_file = outputpath+"config.yaml"
with open(output_config_file, "w") as file:
    yaml.dump(config, file)
    print(f"Saved configuration parameters to '{output_config_file}'.")
print ("=====================================")

outfile=ROOT.TFile(outputpath+fileOutName,"RECREATE")
for key in hRecoEventMuonP_nLeptons.keys():
  hRecoEventMuonP_nLeptons[key].Write()
  hRecoEventElectronP_nLeptons[key].Write()
  hRecoEventTauP_nLeptons[key].Write()


hRecoEventDist.Write()
hRecoEventTypeDist.Write()
hRecoEventMuonP.Write()
hRecoEventElectronP.Write()
hRecoEventTauP.Write()
hRecoPairCharge.Write()
hRecoEventCharge.Write()
outfile.Close() 
print ("Plots saved in %s" %outputpath+fileOutName)
print ("=====================================")


