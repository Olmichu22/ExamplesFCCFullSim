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
parser.add_argument("-R", "--dRMax", default=0.4, type=float, help="Maximum delta R value")
parser.add_argument("-n", "--NeutronCut", default=1, type=float, help="Neutron momentum cut value")

args = parser.parse_args()

minPTau=args.TauPCut
minPMuon=args.ElectPCut
minPElectron=args.MuonPCut

# file = ROOT.TFile("Event_dist_Event_dist_decayall_0.4_t0.1_m0.5_e0.5_n1.root")
inputpath = f"Results/ZReco/Event_dist_decayAll_{args.dRMax}_t{args.TauPCut}_m{args.MuonPCut}_e{args.ElectPCut}_n{args.NeutronCut}"

outputpath = f"Results/ZReco/Event_dist_decayAll_{args.dRMax}_t{args.TauPCut}_m{args.MuonPCut}_e{args.ElectPCut}_n{args.NeutronCut}/Images"
if not os.path.exists(outputpath):
    os.makedirs(outputpath)

decayString = f"Event_dist_decayAll_{args.dRMax}_t{args.TauPCut}_m{args.MuonPCut }_e{args.ElectPCut}_n{args.NeutronCut}"
file_path = inputpath+"/"+decayString+".root"


variabs = ["histRecoNLeptonsEventDist",
           "histRecoEventMuonP",
           "histRecoEventElectronP",
           "histRecoEventTauP",
           "histRecoPairCharge",
           "histRecoCharge"]

plot_titles_config = {"histRecoNLeptonsEventDist": {"x": "Number of Leptons", "y":"Events", "title":"Histogram of Number of Leptons per Event"},
                      "histRecoEventMuonP": {"x": "Reco Muon P", "y":"Events", "title":"Histogram of Reco Muon P"},
                      "histRecoEventElectronP": {"x": "Reco Electron P", "y":"Events", "title":"Histogram of Reco Electron P"},
                      "histRecoEventTauP": {"x": "Reco Tau P", "y":"Events", "title":"Histogram of Reco Tau P"},
                      "histRecoPairCharge": {"x": "Charge", "y":"Events", "title":"Histogram of Total Charge in Reco dual Leptons"},
                      "histRecoCharge": {"x": "Charge", "y":"Events", "title":"Histogram of Reco Charge"}
                      }


colors = ["kBlack", "kRed", "kGreen", "kBlue", "kYellow", "kMagenta", "kCyan", "kOrange", "kSpring", "kTeal", "kAzure", "kViolet", "kPink"]
colors = colors=[getattr(ROOT,colors[i]) for i in range(0,len(colors))]
# Read nLeptons values from a text file
nLeptons_file = inputpath+"/"+decayString+"_keys.txt"
with open(nLeptons_file, "r") as file:
  nLeptons_values = [int(line.strip()) for line in file.readlines()]

# Same Hist variabs
same_hist_variabs = {"electron":[], "muon": [], "tau": []}

# Add histograms for each nLeptons value
for i, nLeptons in enumerate(nLeptons_values):
  hist_name_electron = f"histRecoEventElectronP_{nLeptons}Leptons"
  hist_name_muon = f"histRecoEventMuonP_{nLeptons}Leptons"
  hist_name_tau = f"histRecoEventTauP_{nLeptons}Leptons"

  # Add histogram names to variabs
  same_hist_variabs["electron"].append(hist_name_electron)
  same_hist_variabs["muon"].append(hist_name_muon)
  same_hist_variabs["tau"].append(hist_name_tau)
  

  # Add plot titles to plot_titles_config
  plot_titles_config[hist_name_electron] = {"x": "Reco Electron P", "y": "Events", "title": f"Histogram of Reco Electron P ({nLeptons} Leptons)", "color": colors[i]}
  plot_titles_config[hist_name_muon] = {"x": "Reco Muon P", "y": "Events", "title": f"Histogram of Reco Muon P ({nLeptons} Leptons)", "color": colors[i]}
  plot_titles_config[hist_name_tau] = {"x": "Reco Tau P", "y": "Events", "title": f"Histogram of Reco Tau P ({nLeptons} Leptons)", "color": colors[i]}


file = ROOT.TFile(file_path)
for var in variabs:
  c=ROOT.TCanvas("c"+var)
  leg=ROOT.TLegend(0.75,0.89,0.95,0.75)
  leg.SetFillStyle(0)
  leg.SetFillColor(0)
  leg.SetLineColor(0)
  histo = file.Get(var)
  print(var)
  histo.SetXTitle(plot_titles_config[var]["x"])
  histo.SetYTitle(plot_titles_config[var]["y"])
  histo.SetTitle(plot_titles_config[var]["title"])
  histo.Draw("hist")
  c.SaveAs(outputpath+"/"+var+".png")
  
for key in same_hist_variabs.keys():
  c=ROOT.TCanvas("c"+key)
  leg=ROOT.TLegend(0.75,0.89,0.95,0.75)
  leg.SetFillStyle(0)
  leg.SetFillColor(0)
  leg.SetLineColor(0)
  for i, var in enumerate(same_hist_variabs[key]):
    histo = file.Get(var)
    histo.SetLineColor(plot_titles_config[var]["color"])
    histo.Draw("hist same")
    leg.AddEntry(histo, plot_titles_config[var]["title"], "l")
  histo.SetXTitle(plot_titles_config[var]["x"])
  histo.SetYTitle(plot_titles_config[var]["y"])
  histo.SetTitle(f"Histogram of {key} P for {nLeptons_values[i]} Leptons")
  leg.Draw()
  c.SaveAs(outputpath+"/"+key+".png")