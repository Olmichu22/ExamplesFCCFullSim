#!/usr/bin/env python
import ROOT 
from ROOT import TH1F,TFile

ROOT.gStyle.SetOptStat(0)

#MESON="PION" 
#cuts="0_0.1_5"
#cuts="PION_1GeVtau_01photon"

MESON="#rho (2#gamma)"
cuts="2_0.1_2"

#MESON="#pi"
#cuts="0_0.1_5"
file=TFile("long.root")

#file=TFile("templates_2_0.1_2_DEF_2M.root")

#file=TFile("templates_2_0.1_2_SM_TREE_1M_JANUARY.root")
#file=TFile("templates_2_0.1_2_SM_1Mevts.root")
#fileOut=TFile("BINED_templates_2_0.1_2_SM_1000_20_GEN_JANUARY.root","RECREATE")
fileOut=TFile("BINED_templates_2_0.1_2_SM_LONG_20_MARCH.root","RECREATE")

#var="OmegaCosThetaTau_GEN"
#title="#omega_{#rho} (GEN)"

histo_events=file.Get("hEvents")
NGEN=histo_events.GetBinContent(1) #1e6 # 976000+9.99e5 
xsec=1476.58*1000 # 1476.58 pb 
lumi=NGEN/ xsec #     17. * 1000. fb-1 # 17 ab-1

scale=1

var="OmegaCosTheta"
title="#omega_{#rho}"

rebin=1

sample=["SIGNAL","SIGNAL_P1","SIGNAL_M1","BG"]
color=[ROOT.kBlack,ROOT.kGreen+2,ROOT.kRed,ROOT.kYellow]
sampleName=["SM","A_{#tau}=+1","A_{#tau}=-1","BG"]

nBins=20  
binLength=100/nBins

for bin in range(0,nBins):

  c=ROOT.TCanvas("canvas","",800,800)
  leg=ROOT.TLegend(0.5,0.89,0.9,0.6)
  leg.SetFillStyle(0)
  leg.SetLineColor(0)
  leg.SetLineWidth(0)

  
  binIni=int(bin*binLength+1)
  binEnd=int((bin+1)*binLength)
  print (bin,binIni,binEnd)

  maxY=0

  histo_bin={}

  for i in range(0,len(sample)):
    print (var+"_"+sample[i])
    histo2D=file.Get(var+"_"+sample[i])
    print ("histo_"+sample[i]+"_"+str(bin)) 
    histo_bin[i]=histo2D.ProjectionX("histo_"+sample[i]+"_"+str(bin),binIni,binEnd)
    histo_bin[i].Rebin(rebin)
    histo_bin[i].Scale(scale)
    histo_bin[i].SetXTitle(title)  
    histo_bin[i].SetLineWidth(2)
    histo_bin[i].SetLineColor(color[i])

    if sample[i]=="BG":
      histo_bin[i].SetFillColor(color[i])
      #histo_bin[i].SetFillStyle(3004)
      leg.AddEntry(histo_bin[i],sampleName[i],"f")
    else: 
      leg.AddEntry(histo_bin[i],sampleName[i],"l")

    print("Integral: ",sampleName, histo_bin[i].Integral())

    if maxY<histo_bin[i].GetMaximum():
      maxY=histo_bin[i].GetMaximum()
      
    if i==0:
      histo_bin[i].Draw("hist")
    else:
      histo_bin[i].Draw("hist,same")

  histo_bin[0].SetMaximum(maxY*1.2)
  leg.Draw()
  c.Draw()
  c.SaveAs("/nfs/cms/cepeda/FCC/plotsJanuary/BINS/BIN_"+var+"_"+cuts+"_"+str(binIni)+"_"+str(binEnd)+".png")

  fileOut.cd()
  for i in range(0,len(sample)):
    histo_bin[i].Write()
  



# Also for the full range
c = ROOT.TCanvas("canvas_full", "", 800, 800)
leg = ROOT.TLegend(0.5, 0.89, 0.9, 0.6)
leg.SetFillStyle(0)
leg.SetLineColor(0)
leg.SetLineWidth(0)

maxY = 0

histo_full = {}

for i in range(0, len(sample)):

  histo2D = file.Get(var + "_" + sample[i])
  print("histo_" + sample[i] + "_full")
  histo_full[i] = histo2D.ProjectionX("histo_" + sample[i] + "_full", 1, 100)
  histo_full[i].Rebin(rebin)
  histo_full[i].Scale(scale)
  histo_full[i].SetXTitle(title)
  histo_full[i].SetLineWidth(2)
  histo_full[i].SetLineColor(color[i])

  if sample[i] == "BG":
    histo_full[i].SetFillColor(color[i])
    # histo_full[i].SetFillStyle(3004)
    leg.AddEntry(histo_full[i], sampleName[i], "f")
  else:
    leg.AddEntry(histo_full[i], sampleName[i], "l")

  print("Integral: ", sampleName, histo_full[i].Integral())

  if maxY < histo_full[i].GetMaximum():
    maxY = histo_full[i].GetMaximum()

  if i == 0:
    histo_full[i].Draw("hist")
  else:
    histo_full[i].Draw("hist,same")

histo_full[0].SetMaximum(maxY * 1.2)
leg.Draw()
c.Draw()
c.SaveAs("/nfs/cms/cepeda/FCC/plotsJanuary/BINS/BIN_" + var + "_" + cuts + "_full.png")

fileOut.cd()
for i in range(0, len(sample)):
  histo_full[i].Write()


file.Close()
