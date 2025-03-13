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

ROOT.gStyle.SetOptStat(0)

parser = argparse.ArgumentParser(description="Configure the analysis",
                         formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("-f", "--filepath", default="ZTauTau_SMPol_25Sept_MuonFix", help="Sample file name to process")
parser.add_argument("-p", "--TauPCut", default=0.1, type=float, help="Tau momentum cut value")
parser.add_argument("-m", "--MuonPCut", default=0.1, type=float, help="Electron momentum cut value")
parser.add_argument("-e", "--ElectPCut", default=0.1, type=float, help="Muon momentum cut value")
parser.add_argument("-R", "--dRMax", default=0.4, type=float, help="Maximum delta R value")
parser.add_argument("-n", "--NeutronCut", default=1, type=float, help="Neutron momentum cut value")


args = parser.parse_args()

minPTau=args.TauPCut
minPMuon=args.ElectPCut
minPElectron=args.MuonPCut
minNeutron=args.NeutronCut
dRmax=args.dRMax

results_path = "Results/RecoZEffficiency"
input_path = f"RecoZtau_decayAll_{args.dRMax}_t{args.TauPCut}_m{args.MuonPCut}_e{args.ElectPCut}_n{args.NeutronCut}"
input_path = f"{results_path}/{input_path}"


file = f"firstTest_RecoZtau_decayAll_{args.dRMax}_t{args.TauPCut}_m{args.MuonPCut}_e{args.ElectPCut}_n{args.NeutronCut}.root"

file = ROOT.TFile(input_path+"/"+file)


outputpath = results_path+"/Images"
if not os.path.exists(outputpath):
    os.makedirs(outputpath)




variabs = [ 
"histoGenZP", 
"histoGenZP",
"histoRecoZP",
"histoGenZMass",
"histoZMass", 
"histoRecoZMass", 
"histoGenTausDeltaTheta",
"histoRecoTausDeltaTheta",
"histoGenZTausType", 
"histoRecoZTausType",
"EffiGenTausType",
"EffiGenZP",
"EffiGenVisZP",
"EffiGenVisMass",
"EffiGenTausDeltaTheta"]

for var in variabs:
  c=ROOT.TCanvas("c"+var)
  leg=ROOT.TLegend(0.75,0.89,0.95,0.75)
  leg.SetFillStyle(0)
  leg.SetFillColor(0)
  leg.SetLineColor(0)
  histo = file.Get(var)
  histo.Draw("hist")
  c.SaveAs(outputpath+"/"+var+".png")