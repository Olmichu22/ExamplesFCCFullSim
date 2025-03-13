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

import argparse
parser = argparse.ArgumentParser(description="Configure the analysis",
                         formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("-f", "--sample", default="ZTauTau_SMPol_25Sept_MuonFix", help="Sample file name to process")
parser.add_argument("-o", "--outfile", default="Event_dist_", help="Output file name prefix")
parser.add_argument("-d", "--decay", default=-777, type=int, help="Decay mode to select (-777 for all)")
parser.add_argument("-p", "--TauPCut", default=0.1, type=float, help="Tau momentum cut value")
parser.add_argument("-m", "--MuonPCut", default=0.1, type=float, help="Electron momentum cut value")
parser.add_argument("-e", "--ElectPCut", default=0.1, type=float, help="Muon momentum cut value")

parser.add_argument("-R", "--dRMax", default=0.4, type=float, help="Maximum delta R value")
parser.add_argument("-n", "--neutronCut", default=1, type=float, help="Neutron momentum cut value")
parser.add_argument("-t", "--test", default="True", type=str, help="Run in test mode with limited number of files")

args = parser.parse_args()
config = vars(args)
print(config)

dRMax=args.dRMax
minPTau=args.TauPCut
minPMuon=args.MuonPCut
minPElectron=args.ElectPCut
selectDecay=args.decay
fileOutName=args.outfile
PNeutron=args.neutronCut
selectDecay=args.decay
sample=args.sample
test= True if args.test=="True" else False

decayString=f"Event_dist_decay{selectDecay}_{dRMax}_t{args.TauPCut}_m{args.MuonPCut }_e{args.ElectPCut}_n{PNeutron}"
if selectDecay==-777:
    decayString=f"Event_dist_decayall_{dRMax}_t{args.TauPCut}_m{args.MuonPCut }_e{args.ElectPCut}_n{PNeutron}"
fileOutName=args.outfile+decayString+".root"

print ("=====================================")

# get all the files 
path="/pnfs/ciemat.es/data/cms/store/user/cepeda/FCC/FullSim/"
file="out_reco_edm4hep_edm4hep"
filenames=[]
dir_path=path+"/"+sample
names = ROOT.std.vector('string')()
nfiles=len(os.listdir(dir_path))

outputpath = f"Images/ZReco/TCut {minPTau} ECut {minPElectron} MCut {minPMuon}"
if not os.path.exists(outputpath):
  os.makedirs(outputpath)

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
hRecoEventMuonP = TH1F("histRecoCardEventMuonP", "Muon Momentum", 50, 0, 50)
hRecoEventElectronP = TH1F("histRecoCardEventElectronP", "Electron Momentum", 50, 0, 50)
hRecoEventTauP = TH1F("histRecoCardEventTauP", "Tau Momentum", 50, 0, 50)
hRecoPairCharge = TH1F("histRecoCardEventOpositePairCharge", "Oposite Charge", 10, -5, 5)
hRecoEventCharge = TH1F("histRecoCardEventOpositeCharge", "Oposite Charge", 10, -5, 5)

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
  recoTaus= tauReco.findAllTaus(pfos,dRMax, minPTau,PNeutron)
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
  recoLeptons = {}
  
  for i, muon in recoMuons.items():
    recoLeptons[f"muon{i}"] = muon
    hRecoEventMuonP.Fill(muon.getMomentum().P())  
  for i, electron in recoElectrons.items():
    recoLeptons[f"electron{i}"] = electron
    hRecoEventElectronP.Fill(electron.getMomentum().P())  
    
  for i, tau in recoTaus.items():
    if tau.getID() == -13 or tau.getID() == -11:
      continue
    recoLeptons[f"tau{i}"] = tau
    hRecoEventTauP.Fill(tau.getMomentum().P())  

  
  # fill histograms depending on the number of leptons
  if nLeptons == 2:
    tot_charge = 0
    pair_cases_count += 1
    for i, lepton in recoLeptons.items():
      tot_charge += lepton.getCharge()
      #Key code for the pair
      #Key code is the key i without the number
      # if tot_charge == 0:
      #   hRecoEventTypeDist.Fill(-1)
    pair_cases_charge[tot_charge] += 1
    hRecoPairCharge.Fill(tot_charge)
    # hRecoEventTypeDist.Fill(pair_cases[key_code])
    
    
  else:
    tot_charge = 0
    for i, lepton in recoLeptons.items():
      tot_charge += lepton.getCharge()
    hRecoEventCharge.Fill(tot_charge)
    
  hRecoEventDist.Fill(nLeptons)

max_case = len(pair_cases)
hRecoEventTypeDist = TH1F("histRecoCardEventTypeDist", "Type of Leptons per Event", max_case, 0, max_case)
event_type = pd.DataFrame(columns=["case", "number"])
event_type["case"] = pair_cases.keys()
event_type["number"] = pair_cases.values()

pair_cases_charge_df = pd.DataFrame(columns=["charge", "number"])
pair_cases_charge_df["charge"] = pair_cases_charge.keys()
pair_cases_charge_df["number"] = pair_cases_charge.values()
# New row with total not using append
pair_cases_charge_df.loc[-1] = ["total", pair_cases_charge_df["number"].sum()]
pair_cases_charge_df.index = pair_cases_charge_df.index + 1
pair_cases_charge_df.sort_index(inplace=True)
pair_cases_charge_df.to_csv(outputpath+"/pair_cases_charge.csv", index=False)

print(f"Parejas totales neutras {pair_cases_charge[0]}")

# Sort by number of cases
event_type = event_type.sort_values(by="number", ascending=False)
event_type["id"] = range(0, len(event_type))
for i, row in event_type.iterrows():
  hRecoEventTypeDist.Fill(row["id"], row["number"])

event_type.to_csv(outputpath+"/event_cases.csv", index=False)

    


  # for j in range(0,nTaus):
  #   recoTauP4=recoTaus[j][0]
  #   recoTauId=recoTaus[j][1]
  #   recoTauQ=recoTaus[j][2]
  #     #recoTauDR=recoTaus[j][3]
  #     #recoTauNConsts=recoTaus[j][4]
  #     #recoTauConsts=recoTaus[j][5]

  #     # to make the code more economic we are checking gen and reco in parallel, but 
  #     # there is a difference in the DM labelling:
  #     # at reco level we count photons and at gen level pi0s: difference in the
  #     # decay mode (1 gen can be 1 or 2 reco, etc )
  #   recoDM=recoTauId
  #   if recoTauId==2:
  #     recoDM=1
  #   elif recoTauId>=3 and recoTauId<10:
  #     recoDM=3
  #   elif (recoTauId>=11 and recoTauId<15):
  #     recoDM=11

  #   if selectDecay!=-777 and selectDecay!=recoDM:
  #     continue


print ("-------------------------------------")
print ("Processed %d events" %countEvents)
# save plots for later
outfile=ROOT.TFile(fileOutName,"RECREATE")

hRecoEventDist.Write()
hRecoEventTypeDist.Write()
hRecoEventMuonP.Write()
hRecoEventElectronP.Write()
hRecoEventTauP.Write()
hRecoPairCharge.Write()
hRecoEventCharge.Write()
outfile.Close() 
print ("Plots saved in %s" %fileOutName)
print ("=====================================")


