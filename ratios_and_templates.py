import ROOT
ROOT.gROOT.SetBatch(True)
ROOT.gStyle.SetOptStat(0)

# Archivos
file_sm = "Results/RhoAnalysis/PolSM_March24_2M_3tau_trained0.4_tph0.35_tpi0_n3_g0.0/Histos_dRgt2.312e-07_inf_tau_traineddecay2_0.4_tph0.35_tpi0_n3_g0.0.root"
file_new = "Results/RhoAnalysis/PolSM_March24_2M_3tau_trained0.4_tph0.35_tpi0_n3_g0.0/Histos_dRgt2.315e-07_inf_tau_traineddecay2_0.4_tph0.35_tpi0_n3_g0.0.root"

f_sm = ROOT.TFile.Open(file_sm, "READ")
f_new = ROOT.TFile.Open(file_new, "READ")

# Histogramas
h_P1_sm = f_sm.Get("Omega_SIGNAL_P1").Clone("h_P1_sm")
h_M1_sm = f_sm.Get("Omega_SIGNAL_M1").Clone("h_M1_sm")
h_P1_new = f_new.Get("Omega_SIGNAL_P1").Clone("h_P1_new")
h_M1_new = f_new.Get("Omega_SIGNAL_M1").Clone("h_M1_new")

h_P1_sm.SetDirectory(0)
h_M1_sm.SetDirectory(0)
h_P1_new.SetDirectory(0)
h_M1_new.SetDirectory(0)

# Rebin
for h in [h_P1_sm, h_M1_sm, h_P1_new, h_M1_new]:
    h.Rebin(2)

# ============ PLOT 1: Comparación de Templates ============
c1 = ROOT.TCanvas("c1", "Templates Comparison", 800, 600)
c1.SetLeftMargin(0.12)
c1.SetRightMargin(0.05)

# Estilos
h_P1_sm.SetLineColor(ROOT.kGreen+4)
h_P1_sm.SetLineStyle(10)
h_P1_sm.SetLineWidth(3)

h_M1_sm.SetLineColor(ROOT.kGreen+4)
h_M1_sm.SetLineStyle(7)
h_M1_sm.SetLineWidth(3)

h_P1_new.SetLineColor(ROOT.kRed)
h_P1_new.SetLineStyle(10)
h_P1_new.SetLineWidth(3)

h_M1_new.SetLineColor(ROOT.kRed)
h_M1_new.SetLineStyle(7)
h_M1_new.SetLineWidth(3)

# Encontrar máximo
max_val = max(h.GetMaximum() for h in [h_P1_sm, h_M1_sm, h_P1_new, h_M1_new])
h_P1_sm.SetMaximum(max_val * 1.2)
h_P1_sm.SetMinimum(0)

h_P1_sm.SetTitle("Templates Comparison #omega (#rho decay)")
h_P1_sm.GetXaxis().SetTitle("#omega")
h_P1_sm.GetYaxis().SetTitle("Events")

h_P1_sm.Draw("HIST")
h_M1_sm.Draw("HIST SAME")
h_P1_new.Draw("HIST SAME")
h_M1_new.Draw("HIST SAME")

leg1 = ROOT.TLegend(0.55, 0.55, 0.9, 0.88)
leg1.SetBorderSize(0)
leg1.SetFillStyle(0)
leg1.SetTextSize(0.03)
leg1.AddEntry(h_P1_sm, "Template P1, sin^{2}#theta_{eff}=0.2312", "l")
leg1.AddEntry(h_M1_sm, "Template M1, sin^{2}#theta_{eff}=0.2312", "l")
leg1.AddEntry(h_P1_new, "Template P1, sin^{2}#theta_{eff}=0.2315", "l")
leg1.AddEntry(h_M1_new, "Template M1, sin^{2}#theta_{eff}=0.2315", "l")
leg1.AddEntry(ROOT.nullptr, "7 fb^{-1}  #sqrt{s}=91GeV", "")
leg1.Draw()

c1.SaveAs("templates_comp_sin_eff.pdf")
c1.SaveAs("templates_comp_sin_eff.png")
print("Guardado: templates_comp_sin_eff.pdf/png")

# ============ PLOT 2: Ratio sin0.2315 / sin0.2312 para P1 y M1 ============
c2 = ROOT.TCanvas("c2", "Ratio sin_eff", 800, 600)
c2.SetLeftMargin(0.12)
c2.SetRightMargin(0.05)

# Ratio P1(0.2315) / P1(0.2312)
h_ratio_P1 = h_P1_new.Clone("h_ratio_P1")
h_ratio_P1.Divide(h_P1_sm)

# Ratio M1(0.2315) / M1(0.2312)
h_ratio_M1 = h_M1_new.Clone("h_ratio_M1")
h_ratio_M1.Divide(h_M1_sm)

h_ratio_P1.SetLineColor(ROOT.kBlue)
h_ratio_P1.SetLineStyle(1)
h_ratio_P1.SetLineWidth(3)

h_ratio_M1.SetLineColor(ROOT.kRed)
h_ratio_M1.SetLineStyle(1)
h_ratio_M1.SetLineWidth(3)

h_ratio_P1.SetMaximum(1.004)
h_ratio_P1.SetMinimum(0.996)

h_ratio_P1.SetTitle("Ratio sin^{2}#theta_{eff}=0.2315 / sin^{2}#theta_{eff}=0.2312")
h_ratio_P1.GetXaxis().SetTitle("#omega")
h_ratio_P1.GetYaxis().SetTitle("Ratio")

h_ratio_P1.Draw("HIST")
h_ratio_M1.Draw("HIST SAME")

# Línea en y=1
line = ROOT.TLine(h_ratio_P1.GetXaxis().GetXmin(), 1, h_ratio_P1.GetXaxis().GetXmax(), 1)
line.SetLineColor(ROOT.kBlack)
line.SetLineStyle(2)
line.Draw()

leg2 = ROOT.TLegend(0.55, 0.7, 0.9, 0.88)
leg2.SetBorderSize(0)
leg2.SetFillStyle(0)
leg2.SetTextSize(0.035)
leg2.AddEntry(h_ratio_P1, "P1: 0.2315 / 0.2312", "l")
leg2.AddEntry(h_ratio_M1, "M1: 0.2315 / 0.2312", "l")
leg2.AddEntry(ROOT.nullptr, "7 fb^{-1}  #sqrt{s}=91GeV", "")
leg2.Draw()

c2.SaveAs("ratio_sin_eff_P1_M1.pdf")
c2.SaveAs("ratio_sin_eff_P1_M1.png")
print("Guardado: ratio_sin_eff_P1_M1.pdf/png")

f_sm.Close()
f_new.Close()
print("Done!")