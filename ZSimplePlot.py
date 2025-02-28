#!/usr/bin/env python
import ROOT 
from ROOT import TH1F 
import argparse

parser = argparse.ArgumentParser(description="Configure the plot",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("-g","--tag",default="0.4_0.1_1")
parser.add_argument("-f","--samples",default=["decayAll"], nargs='+', type=str) #,"decay0","decay1","decay10"]
parser.add_argument("-l","--labels",default=["all taus"], nargs='+', type=str) #, "#pi", "#rho","a_{1} (3#pi)"]
parser.add_argument("-v","--variabs",default=["histoGenZP"], nargs='+', type=str) 
parser.add_argument("-c","--colors",default=["kBlack"], nargs='+', type=str) # ,ROOT.kRed,ROOT.kBlue,ROOT.kGreen+2]

#variabs=["histoRecoTauType","histoRecoTauP","histoRecoTauMass","histoRecoTauTheta","histoGenTauP","histoGenTauType","histoGenTauVisP","histoGenTauTheta","histoGenTauVisMass"]
#xLabels=["Tau Type, Reco","Reco Tau P (GeV)","Reco Tau Mass (GeV)","Tau Theta","Gen Tau P","Gen Tau Type","Gen Tau Visible P (GeV)","Gen Tau Theta","Gen Tau Visible Mass"]

xLabels_dict = {"histoGenZP":"Gen Z P (GeV)",
                "histoGenVisZP":"Gen Z Visible P (GeV)",
                "histoGenZMass":"Gen Z Mass (GeV)",
                "histoGenVisZMass":"Gen Z Visible Mass (GeV)",
                "histoGenTausTheta":"Gen Z Taus Angle (Rad)",
                "histoGenZTausType":"Gen Z Taus Decay Type"}

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
  files[i]=ROOT.TFile("firstTest_Ztau_"+samples[i]+"_"+tag+".root")

#Format for the rate histograms:
def formatHisto(file,variab,rename,titleX,color=ROOT.kBlack):
  histo = file.Get(variab)
  histo.SetName(rename)
  histo.SetXTitle(titleX)
  histo.SetLineColor(color)
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
    histo[i]=formatHisto(files[i],var,samples[i]+var,xLabels[iv],colors[i])
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
    histo[0].GetXaxis().SetRangeUser(0,100)

  if "Type" in var:
  #   histo[0].GetYaxis().SetRangeUser(0, 1.3)
  # #   histo[0].SetMinimum(1)
    histo[0].GetYaxis().SetTitle("Frecuency")
  else:
    histo[0].GetYaxis().SetTitle("Counts")

  histo[0].SetMaximum(max*1.4)

  iv=iv+1
  leg.Draw()
  c.SaveAs(var+".png")
  # Keep graph open
  
  # c.SetLogy()
  # c.SaveAs(var+"_LOG.png")
