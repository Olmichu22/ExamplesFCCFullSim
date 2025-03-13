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

parser.add_argument("-p", "--TauPCut", default=0.1, type=float, help="Tau momentum cut value")
parser.add_argument("-m", "--MuonPCut", default=0.1, type=float, help="Electron momentum cut value")
parser.add_argument("-e", "--ElectPCut", default=0.1, type=float, help="Muon momentum cut value")


file = ROOT.TFile("Event_dist_Event_dist_decayall_0.4_t0.1_m0.5_e0.5_n1.root")

minPTau=parser.TauPCut
minPMuon=parser.ElectPCut
minPElectron=parser.MuonPCut
outputpath = f"Images/ZReco/TCut {parser.TauPCut} ECut {parser.ElectPCut} MCut {parser.MuonPCut}"

variabs = ["histRecoCardEventDist",
           "histRecoCardEventMuonP",
           "histRecoCardEventElectronP",
           "histRecoCardEventTauP"]

for var in variabs:
  c=ROOT.TCanvas("c"+var)
  leg=ROOT.TLegend(0.75,0.89,0.95,0.75)
  leg.SetFillStyle(0)
  leg.SetFillColor(0)
  leg.SetLineColor(0)
  histo = file.Get(var)
  histo.Draw("hist")
  c.SaveAs(outputpath+"/"+var+".png")