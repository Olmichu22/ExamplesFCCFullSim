#! /usr/bin/env python3

import ROOT
from ROOT import TFile, TH1F, TCanvas, TMinuit, TLegend, gStyle, gPad
import ctypes
import math 
import numpy as np
import sys
# Options
force_perfect_agreement =  False
use_likelihood = True

tag="_def"
if force_perfect_agreement: tag="_perfect"

#filename="BINED_templates_2_0.1_2_SM_1000_20_FREEZE.root"
#filename="BINED_templates_2_0.1_2_SM_1000_20_JANUARY_FREEZE.root"
#BINED_templates_2_0.1_2_SM_1000_20_GEN_JANUARY.root"

#filename="BINED_templates_2_0.1_2_SM_2M_20_MARCH.root"
#filename="BINED_templates_2_0.1_2_SM_1000_20_JANUARY_realscale.root"
#filename="BINED_templates_2_0.1_2_SM_1000_20_MARCH.root"

filename="Binned_histograms/BINED_template_Histos_dRgt3.06_5.0_MesonPgt0.0_lt45.57_LeptonPgt0.0_lt41.16_tau_traineddecay2_0.4_tph0.35_tpi0_n3_g0.0_complete.root"

filenameP1=filename

nBins=20
binSize=100./nBins
rebin=2

NGEN=1e6 # 976000+9.99e5 
xsec=1476.58*1000 # en fb
lumi=5.2 # fb-1  17. * 1000.  # 17 ab-1
scale= 1 # 17000/lumi #*xsec/NGEN # already done in pintarAnalysisBines.py

var="histo"
title="#omega_{#rho}"


file=ROOT.TFile.Open(filename)

binName= "_1"

vectorPol={}
vectorPolError={}
vectorBins={}
vectorCosThetaMin={}
vectorCosThetaMax={}

#for bin in range(0,10):
def poly_fit(X, y, degree):
    """Perform a polynomial fit of given degree to the data points (X, y)."""
    coeffs = np.polyfit(X, y, degree)
    return coeffs

def dofit(bin, fullRange=False):

    binName="_full"

    if fullRange==False:
        binName="_"+str(bin)
        vectorBins[bin]=bin
        vectorCosThetaMin[bin]=bin*binSize/100*2 -1 
        vectorCosThetaMax[bin]=(bin+1)*binSize/100*2 -1
    else: 
        binName="_full"
        vectorBins[bin]=nBins
        vectorCosThetaMin[bin]=-1
        vectorCosThetaMax[bin]=1

    hist_bg  = file.Get(var+"_BG"+binName)   # Background template
    hist_m1  = file.Get(var+"_SIGNAL_M1"+binName)   # Signal template m1
    hist_p1  = file.Get(var+"_SIGNAL_P1"+binName)   # Signal template p1
    hist_sm  = file.Get(var+"_SIGNAL"+binName)   # Signal template m1

    print ("reading ",var+"_SIGNAL"+binName)

    if force_perfect_agreement:
        # Force the sume to be "perfect"
        hist_data = hist_bg.Clone()
        hist_data.SetName("hist_data")
        ctheta=(vectorCosThetaMax[bin]+vectorCosThetaMin[bin])/2
        Atautheo=0.14955426 #0.150
        Aetheo=  0.14955426 # 0.150
        perfectPtau=-(Atautheo*(1+ctheta*ctheta)+2*Aetheo*ctheta) / (1+ctheta*ctheta + 2*Aetheo*Atautheo*ctheta)
        hist_data.Add(hist_p1,(1.+perfectPtau)/2) 
        hist_data.Add(hist_m1,(1.-perfectPtau)/2)
    else:
        # Use histogram ALL from generation
        hist_data  = hist_sm.Clone()   # generated SM
        hist_data.SetName("hist_data")
        hist_data.Add(hist_bg) 

    hist_data.Rebin(rebin)
    hist_bg.Rebin(rebin)
    hist_p1.Rebin(rebin)
    hist_m1.Rebin(rebin)
    hist_sm.Rebin(rebin)

    print("hist_bg entries: %s, hist_m1 entries: %s, hist_p1 entries %s" % (hist_bg.Integral(),hist_m1.Integral(),hist_p1.Integral()))

    hist_data.Scale(scale)
    hist_bg.Scale(scale) # fixed

    # Ensure histograms have the same binning
    nbins = hist_data.GetNbinsX()
    if nbins != hist_bg.GetNbinsX() or nbins != hist_m1.GetNbinsX() or nbins != hist_p1.GetNbinsX():
        print("Histograms have different binning. Please rebin to have the same number of bins.")
        sys.exit()

    # Define the chi-squared function for MINUIT
    def my_minimization(npar, gin, f, par, iflag):
        val = 0.0

        startbin=hist_data.GetXaxis().FindBin(-1)
        endbin=hist_data.GetXaxis().FindBin(1.4)
        for i in range(startbin,endbin+1):
            observed = hist_data.GetBinContent(i)

            bg= hist_bg.GetBinContent(i)
            # There is some ambiguity in what I am calling +1 and -1 
            # remember Poltau=-Atau
            Nm1= (1-par[1])/2 * hist_m1.GetBinContent(i)
            Np1= (1+par[1])/2 * hist_p1.GetBinContent(i)

            expected=bg+par[0]*(Nm1+Np1)
            if expected<=0: continue  # Skip bins with zero error

            if use_likelihood: 
                # -2*logL
                val += 2*(expected - observed*math.log(expected))
            else:
                # chi2 
                val += (observed-expected)**2 / expected


        f.value = val  

    # Initialize MINUIT
    minuit = ROOT.TMinuit(3)
    minuit.SetFCN(my_minimization)
    minuit.SetPrintLevel(1)  # Set to 1 for standard output

    # Set initial parameters and limits
    start_vals = [scale, 0.150]  # Initial guesses for param factors [bg_scale, m1_scale, p1_scale]
    step_sizes = [0.01, 0.0001]
    param_names = ["scale", "pol"]

    minuit.DefineParameter(0, param_names[0], start_vals[0], step_sizes[0], 0, 2*scale)
    minuit.DefineParameter(1, param_names[1], start_vals[1], step_sizes[1], -1,1)

    # Perform the minimization
    migrad_result = minuit.Command("MIGRAD")
    if migrad_result != 0:
        print("Minimization did not converge.")
        #sys.exit()

    # Retrieve the fit results using ctypes.c_double()
    par_values = ctypes.c_double(0.0)
    par_errors = ctypes.c_double(0.0)
    param_factors = []
    param_errors = []

    for i in range(2):
        minuit.GetParameter(i, par_values, par_errors)
        param_factors.append(par_values.value)
        param_errors.append(par_errors.value)
        print(i, param_factors[i], param_errors[i])

    # Build the fitted histogram
    hist_m1_scaled = hist_m1.Clone()
    hist_p1_scaled = hist_p1.Clone()
    hist_m1_scaled.Scale(param_factors[0]*(1-param_factors[1])/2)
    hist_p1_scaled.Scale(param_factors[0]*(1+param_factors[1])/2)

    hist_fit_total = hist_bg.Clone()
    hist_fit_total.SetFillColor(0) 
    hist_data.SetFillColor(0) 
    hist_data.SetLineColor(ROOT.kBlack)

    hist_fit_total.Add(hist_m1_scaled)
    hist_fit_total.Add(hist_p1_scaled)

    # Plotting the results
    gStyle.SetOptStat(0)
    canvas = TCanvas("canvas", "Fit Result", 800, 600)

    hist_data.SetMarkerStyle(20)
    hist_data.SetMarkerSize(1)
    hist_data.SetTitle("Fit Test")
    hist_data.GetXaxis().SetTitle(title)
    hist_data.GetYaxis().SetTitle("Events")
    hist_data.Draw("E,HIST")

    hist_fit_total.SetLineColor(ROOT.kRed)
    hist_fit_total.SetLineWidth(2)
    hist_fit_total.Draw("HIST SAME")

    # Draw the individual components
    colors = [ROOT.kBlue, ROOT.kGreen+2, ROOT.kMagenta]
    styles = [2, 3, 4]

    hist_bg.SetLineColor(colors[0])
    hist_bg.Draw("HIST SAME")

    hist_m1_scaled.SetLineColor(colors[1])
    hist_p1_scaled.SetLineColor(colors[2])
    hist_m1_scaled.Draw("HIST SAME")
    hist_p1_scaled.Draw("HIST SAME")

    print ("m1:",hist_m1_scaled.Integral())
    print ("p1:",hist_p1_scaled.Integral())
    
    # Add legend
    legend = TLegend(0.65, 0.65, 0.88, 0.88)
    legend.SetBorderSize(0)
    legend.SetFillStyle(0)

    legend.AddEntry(hist_data, "Pseudo-data", "lep")
    legend.AddEntry(hist_fit_total, "Total", "l")
    labels = ["Background (fixed)", "Template A_{tau}= +1 ", "Template A_{tau}= -1"]
    legend.AddEntry(hist_bg, labels[0], "l")
    legend.AddEntry(hist_m1_scaled, labels[1], "l")
    legend.AddEntry(hist_p1_scaled, labels[2], "l")
    legend.Draw()

    # Update canvas
    gPad.Update()

    canvas.SaveAs("fit_result_minuit_"+str(bin)+tag+".png")

    #Yields:
    print ("Yields: %s, %s, %s, %s" % (hist_bg.Integral(),hist_m1_scaled.Integral(),hist_p1_scaled.Integral(),hist_fit_total.Integral()))   

    # print numbers
    print("\n>>>> Result P_tau: (%.5f +- %.5f)%% " %(100*param_factors[1],100*param_errors[1] ))
    vectorPol[bin]=param_factors[1] 
    vectorPolError[bin]=param_errors[1]


for bin in range(0,nBins):
    dofit(bin, False)

dofit(nBins, True)


graph = ROOT.TGraphErrors()

for bin in range(0, nBins+1):
    print (" Bin %d, Range CosTheta (%.2f,%.2f),  Result P_tau: (%.5f +- %.5f)%% " %(bin, vectorCosThetaMin[bin], vectorCosThetaMax[bin], vectorPol[bin], vectorPolError[bin]) )
    if bin==nBins: continue
    x = (vectorCosThetaMin[bin] + vectorCosThetaMax[bin]) / 2
    y = vectorPol[bin]
    xerr = (vectorCosThetaMax[bin] - vectorCosThetaMin[bin]) / 2
    yerr = vectorPolError[bin]
    graph.SetPoint(bin, x, y)
    graph.SetPointError(bin, xerr, yerr)

#graph.SetTitle("Polarization vs CosTheta")
graph.GetYaxis().SetTitleOffset(1.4)
graph.GetXaxis().SetTitle("cos #theta_{#tau}")
graph.GetYaxis().SetTitle("P_{#tau}(cos #theta_{#tau})")

canvas2 = TCanvas("canvas2", "Polarization vs CosTheta", 800, 600)
graph.Draw("AP")
graph.SetMarkerStyle(20)
canvas2.SaveAs("polarization_vs_costheta"+tag+".png")
canvas2.SetTitle("")

# Now let's do the second fit to extract Atau and Ae 

# Define the chi-squared function for MINUIT for the second fit
def my_minimization2(npar, gin, f, par, iflag):
    val = 0.0
    for bin in range(0, nBins):
        ctheta = (vectorCosThetaMin[bin] + vectorCosThetaMax[bin]) / 2
        observed = vectorPol[bin]
        observed_err = vectorPolError[bin]
        Atau = par[0]
        Ae = par[1]
        expected = -(Atau * (1 + ctheta * ctheta) + 2 * Ae * ctheta) / (1 + ctheta * ctheta + 2 * Ae * Atau * ctheta)
        if  observed_err != 0:
            val += (observed - expected) ** 2 / observed_err ** 2
    f.value = val

# Initialize MINUIT for the second fit
minuit2 = ROOT.TMinuit(2)
minuit2.SetFCN(my_minimization2)
minuit2.SetPrintLevel(1)  # Set to 1 for standard output

# Set initial parameters and limits for the second fit
start_vals2 = [0.14, 0.16]  # Initial guesses for Atau and Ae
step_sizes2 = [0.00001, 0.00001]
param_names2 = ["Atau", "Ae"]

minuit2.DefineParameter(0, param_names2[0], start_vals2[0], step_sizes2[0], 0, 0.2)
minuit2.DefineParameter(1, param_names2[1], start_vals2[1], step_sizes2[1], 0, 0.2)

# Perform the minimization for the second fit
migrad_result2 = minuit2.Command("MIGRAD")
if migrad_result2 != 0:
    print("Minimization did not converge.")
    #sys.exit()

# Retrieve the fit results using ctypes.c_double()
par_values2 = ctypes.c_double(0.0)
par_errors2 = ctypes.c_double(0.0)
param_factors2 = []
param_errors2 = []

for i in range(2):
    minuit2.GetParameter(i, par_values2, par_errors2)
    param_factors2.append(par_values2.value)
    param_errors2.append(par_errors2.value)

print("\n>>>> Results per bin:")
for bin in range(0, nBins+1):
    print (" Bin %d, Range CosTheta (%.2f,%.2f),  Result P_tau: (%.5f +- %.5f)%% " %(bin, vectorCosThetaMin[bin], vectorCosThetaMax[bin], vectorPol[bin], vectorPolError[bin]) )

# Print the results
print("\n>>>> Full Results:")
print("\n>>>> Result Atau: (%.5f +- %.5f)" % (param_factors2[0]*100, param_errors2[0]*100))
print(">>>> Result Ae: (%.5f +- %.5f)" % (param_factors2[1]*100, param_errors2[1]*100))

# Calculate gv_ga from Ae_sm
Ae_sm = param_factors2[1]
er_Ae_sm= param_errors2[1]
gv_ga = (1 - math.sqrt(1 -  Ae_sm*Ae_sm)) / Ae_sm

term1 = (Ae_sm**2) / math.sqrt(1 - Ae_sm**2)
term2 = 1 - math.sqrt(1 - Ae_sm**2)
d_gv_ga_d_Ae_sm = (term1 - term2) / (Ae_sm**2)
er_gv_ga = abs(d_gv_ga_d_Ae_sm) * er_Ae_sm

print(">>>> Result gv_ga: (%.7f +- %.7f)" % (gv_ga, er_gv_ga))
# Calculate *sin2theta_effective from gv_ga
sin2theta_effective = (1-gv_ga ) / 4
print(">>>> Result sin2theta_effective: (%.7f +- %.7f)" % (sin2theta_effective, er_gv_ga/4))


# Plot the fitted function
fit_func = ROOT.TF1("fit_func", "-([0]*(1+x*x)+2*[1]*x) / (1+x*x + 2*[1]*[0]*x)", -1, 1)
fit_func.SetParameters(param_factors2[0], param_factors2[1])
fit_func.SetLineColor(ROOT.kRed)

canvas2.cd()
fit_func.Draw("SAME")
canvas2.Update()
leg=ROOT.TLegend(0.5,0.89,0.85,0.65)
leg.SetFillStyle(0)
leg.SetLineColor(0)
leg.SetLineWidth(0)
leg.AddEntry("NULL","#sqrt{s}=91 GeV, %4.2f  fb^{-1}" %lumi ,"")
leg.AddEntry(graph,"Pseudo Data","pl")
leg.AddEntry(fit_func,"Fit for A_{#tau} and A_{e}","l")
leg.Draw()
canvas2.SaveAs("polarization_vs_costheta_fit"+tag+".png")
canvas2.SaveAs("polarization_vs_costheta_fit"+tag+".pdf")
canvas2.SaveAs("polarization_vs_costheta_fit"+tag+".C")
file.Close()

