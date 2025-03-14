import sys, os, math 
from array import array
import ROOT
from ROOT import TFile, TTree, TH1F, TH2F
import numpy as np
from podio import root_io
import edm4hep
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt


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
folder_name = f"RecoZtau_decayAll_{args.dRMax}_t{args.TauPCut}_m{args.MuonPCut}_e{args.ElectPCut}_n{args.NeutronCut}"
input_path = f"{results_path}/{folder_name}"


file = f"firstTest_RecoZtau_decayAll_{args.dRMax}_t{args.TauPCut}_m{args.MuonPCut}_e{args.ElectPCut}_n{args.NeutronCut}.root"

file = ROOT.TFile(input_path+"/"+file)


outputpath = input_path+"/Images"
if not os.path.exists(outputpath):
    os.makedirs(outputpath)




variabs = [ 
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

plot_titles_config = {
"histoGenZP": {"x":"Gen Z P (GeV)", "y":"Events", "title":"Histogram of Gen Z P"},
"histoRecoZP": {"x":"Reco Z P (GeV)", "y":"Events", "title":"Histogram of Reco Z P"},
"histoGenZMass": {"x":"Gen Z Mass (GeV)", "y":"Events", "title":"Histogram of Gen Z Mass"},
"histoZMass": {"x":"Z Vis Mass (GeV)", "y":"Events", "title":"Histogram of Z Vis Mass"},
"histoRecoZMass": {"x":"Reco Z Mass (GeV)", "y":"Events", "title":"Histogram of Reco Z Mass"},
"histoGenTausDeltaTheta": {"x":"Gen Taus Delta Theta", "y":"Events", "title":"Histogram of Gen Taus Delta Theta"},
"histoRecoTausDeltaTheta": {"x":"Reco Taus Delta Theta", "y":"Events", "title":"Histogram of Reco Taus Delta Theta"},
"histoGenZTausType": {"x":"Gen Z Taus Type", "y":"Events", "title":"Histogram of Gen Z Taus Type"},
"histoRecoZTausType": {"x":"Reco Z Taus Type", "y":"Events", "title":"Histogram of Reco Z Taus Type"},
"EffiGenTausType": {"x":"Gen Taus Type", "y":"Efficiency", "title":"Histogram of Gen Taus Type"},
"EffiGenZP": {"x":"Gen Z P (GeV)", "y":"Efficiency", "title":"Histogram of Gen Z P"},
"EffiGenVisZP": {"x":"Gen Z Visible P (GeV)", "y":"Efficiency", "title":"Histogram of Gen Z Visible P"},
"EffiGenVisMass": {"x":"Gen Z Visible Mass (GeV)", "y":"Efficiency", "title":"Histogram of Gen Z Visible Mass"},
"EffiGenTausDeltaTheta": {"x":"Gen Taus Delta Theta", "y":"Efficiency", "title":"Histogram of Gen Taus Delta Theta"}}

for var in variabs:
  c=ROOT.TCanvas("c"+var)
  leg=ROOT.TLegend(0.75,0.89,0.95,0.75)
  leg.SetFillStyle(0)
  leg.SetFillColor(0)
  leg.SetLineColor(0)
  histo = file.Get(var)
  histo.SetXTitle(plot_titles_config[var]["x"])
  histo.SetYTitle(plot_titles_config[var]["y"])
  histo.SetTitle(plot_titles_config[var]["title"])
  histo.Draw("hist")
  c.SaveAs(outputpath+"/"+var+".png")
  
# CM Matrix
cm_matrix_path = f"{input_path}/ConfusionMatrix.csv"
cm_matrix = pd.read_csv(cm_matrix_path)

key_to_label = {0: "muonmuon", 1: "muonelectron", 2: "muontau", 3: "electronelectron", 4: "electrontau", 5: "tautau"}

# Plot the confusion matrix
fig, ax = plt.subplots()
im = ax.imshow(cm_matrix.values)
# Put values in the matrix
for i in range(len(cm_matrix.columns)):
    for j in range(len(cm_matrix.columns)):
        text = ax.text(j, i, cm_matrix.values[i, j], ha="center", va="center", color="w")
# Set the labels
ax.set_xlabel("Predicted")
ax.set_ylabel("True")
# Rotate the tick labels and set their alignment.
plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
# We want to show all ticks...
ax.set_xticks(np.arange(len(cm_matrix.columns)))
ax.set_yticks(np.arange(len(cm_matrix.columns)))
# ... and label them with the respective list entries
ax.set_xticklabels([key_to_label[i] for i in range(len(cm_matrix.columns))])
ax.set_yticklabels([key_to_label[i] for i in range(len(cm_matrix.columns))])
# Save the figure
plt.savefig(f"{outputpath}/ConfusionMatrix.png")
plt.close()