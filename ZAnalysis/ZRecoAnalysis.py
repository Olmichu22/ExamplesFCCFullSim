import sys, os, math 
from array import array
import ROOT
from ROOT import TFile, TTree, TH1F, TH2F
import numpy as np
from podio import root_io
import edm4hep
from pathlib import Path
import pandas as pd
from modules import ZReco
from modules import myutils
from sklearn.metrics import confusion_matrix
 

import argparse
parser = argparse.ArgumentParser(description="Configure the analysis",
                         formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("-f", "--sample", default="ZTauTau_SMPol_25Sept_MuonFix", help="Sample file name to process")
parser.add_argument("-o", "--outfile", default="firstTest_", help="Output file name prefix")
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

decayString=f"RecoZtau_decay{selectDecay}_{dRMax}_t{args.TauPCut}_m{args.MuonPCut }_e{args.ElectPCut}_n{PNeutron}"
if selectDecay==-777:
    decayString=f"RecoZtau_decayAll_{dRMax}_t{args.TauPCut}_m{args.MuonPCut }_e{args.ElectPCut}_n{PNeutron}"
fileOutName=args.outfile+decayString+".root"

# get all the files 
path="/pnfs/ciemat.es/data/cms/store/user/cepeda/FCC/FullSim/"
file="out_reco_edm4hep_edm4hep"
filenames=[]
dir_path=path+"/"+sample
nfiles=len(os.listdir(dir_path))

output_dir = f"Results/RecoZEffficiency/{decayString}"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)


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
pfobjects= "PandoraPFOs"

hGenZP = TH1F("histoGenZP", "Gen Z momentum", 50, 0, 100)
hGenVisZP = TH1F("histoGenZP", "Gen Vis Z momentum", 50, 0, 100)
hRecoZP = TH1F("histoRecoZP", "Reco Z momentum", 50, 0, 100)
hMatchedGenZP = TH1F("histoMatchedGenZP","", 50, 0, 100)
hMatchedGenVisZP = TH1F("histoMatchedGenVisZP","", 50, 0, 100)

hGenZMass = TH1F("histoGenZMass", "Gen Z mass", 100, 0, 100)
hGenZVisMass = TH1F("histoZMass", "Gen Vis Z mass", 100, 0, 100)
hRecoZMass = TH1F("histoRecoZMass", "Reco Z mass", 100, 0, 100)
hMatchedGenZVisMass = TH1F("histoMatchedGenZVisMass", "",100, 0, 100)

hGenZTausDeltaTheta = TH1F("histoGenTausDeltaTheta", "Gen Taus Delta theta", 100, 0, 5)
hRecoZTausDeltaTheta = TH1F("histoRecoTausDeltaTheta", "Reco Taus Delta theta", 100, 0, 5)
hMatchedGenZTausDeltaTheta = TH1F("histoMatchedGenVisDeltaTheta","", 100, 0, 5)

hGenZTausType = TH1F("histoGenZTausType", "Gen Z Taus Decay type", 40,-20,20)
hRecoZTausType = TH1F("histoRecoZTausType", "Reco Z Taus Decay type", 40,-20,20)
hMatchedGenZType = TH1F("histoMatchedGenZType", "",40,-20,20)

hGenZTausTypeDict = {}
hRecoZTausTypeDict = {}
hRecoZTausTypeCoincidence = {"Reco":[], "Gen":[]}
recodify_id_dict = {2:1,
                    12:11, 13:11, 14:11,
                    4:3, 5:3, 6:3, 9:3}

print("------------------------------------")
print("Start processing!")

countEvents = 0
for event in reader.get("events"):
  if countEvents%500==0:
      print("... %d" %countEvents)
  countEvents += 1
  # get the constituents
  mc_particles = event.get(genparts)
  pfos = event.get(pfobjects)
  
  # Reco of Z boson at reco level
  # We "know" that there is only one Z
  recoZ = ZReco.findZ(pfos, dRMax, minPTau, minPMuon, minPElectron, PNeutron)
  if recoZ is not None:
    # Maybe this condition should be removed  
    
    # Fill data with reco Z
    hRecoZP.Fill(recoZ.getMomentum().P())
    hRecoZMass.Fill(recoZ.getMass())
    hRecoZTausDeltaTheta.Fill(recoZ.getMaxCone())
    daurecoZ = recoZ.getDaughters()
    
    for ndau in range(len(daurecoZ)):
      ndau_id = daurecoZ[ndau].getID()
    #   if ndau_id < 0:
        #   print(ndau_id)
      try:
          ndau_id = recodify_id_dict[ndau_id]
      except:
          pass
      if ndau_id not in hRecoZTausTypeDict.keys():
        hRecoZTausTypeDict[ndau_id] = 0
      hRecoZTausTypeDict[ndau_id] += 1
      hRecoZTausType.Fill(ndau_id)
      
        
  # Reco of Z boson at generator level
  # It is supposed that there is only one Z
  genZs = ZReco.findAllGenZs(mc_particles)
  nGenZs = len(genZs)
  genZ = genZs[0] if nGenZs==1 else None
  if genZ is not None:
  
  # for i in range(nGenZs):
    genZ = genZs[0]
    hGenZP.Fill(genZ.getMomentum().P())
    hGenVisZP.Fill(genZ.getvisMomentum().P())
    hGenZMass.Fill(genZ.getMass())
    hGenZVisMass.Fill(genZ.getVisMass())
    hGenZTausDeltaTheta.Fill(genZ.getMaxAngle())
    daugenZ = genZ.getDaughters()
    for ndau in range(len(daugenZ)):
        if daugenZ[ndau].getID() not in hGenZTausTypeDict.keys():
            hGenZTausTypeDict[daugenZ[ndau].getID()] = 0
        hGenZTausTypeDict[daugenZ[ndau].getID()] += 1
        hGenZTausType.Fill(daugenZ[ndau].getID())

  # Hard assumption: if we have a reco Z and a gen Z, they are the same
  # even if their constituents are not the same
  if recoZ is not None and genZ is not None:
    hRecoZTausTypeCoincidence["Reco"].append(recoZ.getID())
    hRecoZTausTypeCoincidence["Gen"].append(genZ.getID())
    hMatchedGenZP.Fill(genZ.getMomentum().P())
    hMatchedGenVisZP.Fill(genZ.getvisMomentum().P())
    hMatchedGenZVisMass.Fill(genZ.getVisMass())
    hMatchedGenZTausDeltaTheta.Fill(genZ.getMaxAngle())
    daugenZ = genZ.getDaughters()
    for ndau in range(len(daugenZ)):
        # if daugenZ[ndau].getID() not in hGenZTausTypeDict.keys():
        #     hGenZTausTypeDict[daugenZ[ndau].getID()] = 0
        # hMatchedGenZType[daugenZ[ndau].getID()] += 1
      hMatchedGenZType.Fill(daugenZ[ndau].getID())
    


# Normalization of hGenZTausType (1 over sum of all entries)


# norm = 1/hGenZTausType.GetEntries()
# hGenZTausType.Scale(norm)

total_Gen_taus_decays = 0
total_Reco_taus_decays = 0

for decay in hGenZTausTypeDict.keys():
    total_Gen_taus_decays += hGenZTausTypeDict[decay]
for decay in hRecoZTausTypeDict.keys():
    total_Reco_taus_decays += hRecoZTausTypeDict[decay]


hGenZTausTypedf = pd.DataFrame(hGenZTausTypeDict.items(), columns=["Decay", "Count"])
hGenZTausTypedf["Fraction"] = hGenZTausTypedf["Count"]/total_Gen_taus_decays
hGenZTausTypedf["Fraction"] = hGenZTausTypedf["Fraction"].apply(lambda x: "{:.2%}".format(x))
hGenZTausTypedf = hGenZTausTypedf.sort_values(by="Count", ascending=False)
hGenZTausTypedf.to_csv(output_dir + "/GenZTausDecay.csv", index=False)

hRecoZTausTypedf = pd.DataFrame(hRecoZTausTypeDict.items(), columns=["Decay", "Count"])
hRecoZTausTypedf["Fraction"] = hRecoZTausTypedf["Count"]/total_Reco_taus_decays
hRecoZTausTypedf["Fraction"] = hRecoZTausTypedf["Fraction"].apply(lambda x: "{:.2%}".format(x))
hRecoZTausTypedf = hRecoZTausTypedf.sort_values(by="Count", ascending=False)
hRecoZTausTypedf.to_csv(output_dir + "/RecoZTausDecay.csv", index=False)

hRecoZTausTypeCoincidence = pd.DataFrame(hRecoZTausTypeCoincidence)
hRecoZTausTypeCoincidence.to_csv(output_dir + "/RecoZTausTypeCoincidence.csv", index=False)
# Confusion matrix
cm = confusion_matrix(hRecoZTausTypeCoincidence["Gen"], hRecoZTausTypeCoincidence["Reco"])
cm_df = pd.DataFrame(cm)
cm_df.columns = [str(i) for i in range(6)]
cm_df.index = [str(i) for i in range(6)]
cm_df.to_csv(output_dir + "/ConfusionMatrix.csv", index=False)

hEffiGenZP = hMatchedGenZP.Clone()
hEffiGenZP.Divide(hGenZP)
hEffiGenZP.SetName("EffiGenZP")

hEffiGenVisZP = hMatchedGenVisZP.Clone()
hEffiGenVisZP.Divide(hGenVisZP)
hEffiGenVisZP.SetName("EffiGenVisZP")

hEffiGenVisMass = hMatchedGenZVisMass.Clone()
hEffiGenVisMass.Divide(hGenZVisMass)
hEffiGenVisMass.SetName("EffiGenVisMass")

hEffiGenTausDeltaTheta = hMatchedGenZTausDeltaTheta.Clone()
hEffiGenTausDeltaTheta.Divide(hGenZTausDeltaTheta)
hEffiGenTausDeltaTheta.SetName("EffiGenTausDeltaTheta")

hEffiGenTausType = hMatchedGenZType.Clone()
hEffiGenTausType.Divide(hGenZTausType)
hEffiGenTausType.SetName("EffiGenTausType")




print("------------------------------------")
print("Processed %d events" %countEvents)
print("Plots saved in %s" %fileOutName)
print("====================================")

# Save histograms
outfile = ROOT.TFile(output_dir + "/"+ fileOutName, "RECREATE")
hGenZP.Write()
hGenVisZP.Write()
hRecoZP.Write()
hMatchedGenZP.Write()
hMatchedGenVisZP.Write()

hGenZMass.Write()
hGenZVisMass.Write()
hRecoZMass.Write()
hMatchedGenZVisMass.Write()

hGenZTausDeltaTheta.Write()
hRecoZTausDeltaTheta.Write()
hMatchedGenZTausDeltaTheta.Write()

hGenZTausType.Write()
hRecoZTausType.Write()
hMatchedGenZType.Write()

hEffiGenZP.Write()
hEffiGenVisZP.Write()
hEffiGenVisMass.Write()
hEffiGenTausDeltaTheta.Write()
hEffiGenTausType.Write()

outfile.Close()