#!/usr/bin/env python
import ROOT 
from ROOT import TH1F 
import argparse

parser = argparse.ArgumentParser(description="Configure the plot",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("-g","--tag",default="0.4_0.1_1")
parser.add_argument("-f","--samples",default=["decayAll"], nargs='+', type=str) #,"decay0","decay1","decay10"]
parser.add_argument("-l","--labels",default=["all taus"], nargs='+', type=str) #, "#pi", "#rho","a_{1} (3#pi)"]
parser.add_argument("-v","--variabs",default=["histoRecoTauP"], nargs='+', type=str) 
parser.add_argument("-c","--colors",default=["kBlack"], nargs='+', type=str) # ,ROOT.kRed,ROOT.kBlue,ROOT.kGreen+2]

#variabs=["histoRecoTauType","histoRecoTauP","histoRecoTauMass","histoRecoTauTheta","histoGenTauP","histoGenTauType","histoGenTauVisP","histoGenTauTheta","histoGenTauVisMass"]
#xLabels=["Tau Type, Reco","Reco Tau P (GeV)","Reco Tau Mass (GeV)","Tau Theta","Gen Tau P","Gen Tau Type","Gen Tau Visible P (GeV)","Gen Tau Theta","Gen Tau Visible Mass"]

xLabels_dict = {
  "histoRecoTauType": {"labelx": "Reco Tau Type", "title": "Reco Tau Type"},
  "histoRecoTauP": {"labelx": "Reco Tau P (GeV)", "title": "Reco Tau P (GeV)"},
  "histoRecoTauMass": {"labelx": "Reco Tau Mass (GeV)", "title": "Reco Tau Mass (GeV)"},
  "histoRecoTauTheta": {"labelx": "Reco Tau Theta (rad)", "title": "Reco Tau Theta (rad)"},
  "histoGenTauP": {"labelx": "Gen Tau P", "title": "Gen Tau P"},
  "histoGenTauType": {"labelx": "Gen Tau Type", "title": "Gen Tau Type"},
  "histoGenTauVisP": {"labelx": "Gen Tau Visible P (GeV)", "title": "Gen Tau Visible P (GeV)"},
  "histoGenTauTheta": {"labelx": "Gen Tau Theta (rad)", "title": "Gen Tau Theta (rad)"},
  "histoGenTauVisMass": {"labelx": "Gen Tau Visible Mass", "title": "Gen Tau Visible Mass"},
  "hGenTauP": {"labelx": "Gen Tau P", "title": "Gen Tau P"},
  "hGenVisTauP": {"labelx": "Gen Tau Visible P", "title": "Gen Tau Visible P"},
  "hEffiGenVisTauPt": {"labelx": "Gen Tau Visible P", "title": "Gen Tau Visible P"},
  "hEffiGenTauType": {"labelx": "Gen Tau Type", "title": "Gen Tau Type"},
  "hEffiGenTauTheta": {"labelx": "Gen Tau Theta (rad)", "title": "Gen Tau Theta (rad)"},
  "hEffiGenVisTauMass": {"labelx": "Gen Vis Mass (GeV)", "title": "Gen Vis Mass (GeV)"},
  "histoGenMaxConstAngle": {"labelx": "Gen Max Angle Const (rad)", "title": "Gen Max Angle Const (rad)"},
}

ROOT.gStyle.SetOptStat(0)

args = parser.parse_args()
tag = args.tag
samples = args.samples
labels = args.labels
variabs = args.variabs
xLabels = [xLabels_dict[i] for i in variabs]
colors=[getattr(ROOT,args.colors[i]) for i in range(0,len(args.colors))]


files={}
for i in range(0,len(samples)):
  files[i]=ROOT.TFile("firstTest_"+samples[i]+"_"+tag+".root")

#Format for the rate histograms:
def formatHisto(file,variab,rename,titleX,title,color=ROOT.kBlack):
  histo = file.Get(variab)
  histo.SetName(rename)
  histo.SetXTitle(titleX)
  histo.SetLineColor(color)
  histo.SetTitle(title)
  # histo.SetLineWidth(2)
  #histo.SetMarkerColor(color)
  #histo.SetMarkerStyle(20)
  histo.Sumw2()
  return histo

# One variable per canvas 
iv=0
for var in variabs:
  c=ROOT.TCanvas("c"+var)
  leg=ROOT.TLegend(0.75,0.89,0.95,0.75)
  leg.SetFillStyle(0)
  leg.SetFillColor(0)
  leg.SetLineColor(0)

  histo={}
  for i in range(0,len(samples)):
    histo[i]=formatHisto(files[i],var,samples[i]+var,xLabels[iv]["labelx"],xLabels[iv]["title"],colors[i])
    leg.AddEntry(histo[i],labels[i],"l")

  # hack loop get the maximum right
  histo[0].Draw("hist")
  max=histo[0].GetMaximum()

  for i in range(1,len(samples)):
      histo[i].Draw("hist,sames")
      if histo[i].GetMaximum()>max:
          max=histo[i].GetMaximum()

  # some style tricks 
  if "Mass" in var:
    histo[0].GetXaxis().SetRangeUser(0,2)

  #if "Type" in var:
  #   histo[0].GetXaxis().SetRangeUser(-1,20)
  #   histo[0].SetMinimum(1)

  histo[0].SetMaximum(max*1.4)
  histo[0].GetYaxis().SetTitle("Efficiency")

  iv=iv+1
  leg.Draw()
  c.SaveAs(var+".png")
  # Keep graph open
  
  # c.SetLogy()
  # c.SaveAs(var+"_LOG.png")
