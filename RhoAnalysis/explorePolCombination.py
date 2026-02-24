#!/usr/bin/env python3
import ROOT
import argparse
import os

ROOT.gROOT.SetBatch(True)
ROOT.gStyle.SetOptStat(0)

def draw_one(canvas_name, outdir, tag, hist_sig, hist_p1, hist_m1, sum_scale, pol):
    c = ROOT.TCanvas(canvas_name, "", 800, 600)

    # sin relleno
    for h in [hist_sig, hist_p1, hist_m1]:
        h.SetFillStyle(0)
        h.SetFillColor(0)

    # pesos de polarización
    w_p1 = (1.0 + pol) / 2.0
    w_m1 = (1.0 - pol) / 2.0

    h_sum = hist_p1.Clone("h_sum")
    h_sum.Scale(w_p1)
    h_sum.Add(hist_m1, w_m1)
    h_sum.Scale(sum_scale)
    h_sum.SetFillStyle(0)
    h_sum.SetFillColor(0)

    # colores
    hist_sig.SetLineColor(ROOT.kBlack)
    hist_p1.SetLineColor(ROOT.kGreen+2)
    hist_m1.SetLineColor(ROOT.kMagenta)
    h_sum.SetLineColor(ROOT.kRed)
    h_sum.SetLineWidth(2)

    # rango Y
    y_max = max(
        hist_sig.GetMaximum(),
        hist_p1.GetMaximum(),
        hist_m1.GetMaximum(),
        h_sum.GetMaximum(),
    )
    hist_sig.SetMaximum(1.25 * y_max)
    hist_sig.SetMinimum(0.0)

    hist_sig.Draw("HIST")
    hist_p1.Draw("HIST SAME")
    hist_m1.Draw("HIST SAME")
    h_sum.Draw("HIST SAME")

    leg = ROOT.TLegend(0.55, 0.65, 0.88, 0.88)
    leg.SetFillStyle(0)
    leg.SetBorderSize(0)
    leg.AddEntry(hist_sig, "SIGNAL", "l")
    leg.AddEntry(hist_p1, "SIGNAL_P1", "l")
    leg.AddEntry(hist_m1, "SIGNAL_M1", "l")
    leg.AddEntry(h_sum, f"(wP1*P1+wM1*M1) x {sum_scale}", "l")
    leg.AddEntry(0, f"wP1={w_p1:.3f}, wM1={w_m1:.3f}", "")
    leg.Draw()

    c.SaveAs(os.path.join(outdir, f"plot_signal_p1_m1_{tag}.png"))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--input", required=True, help="ROOT input file")
    ap.add_argument("--nBins", type=int, default=20)
    ap.add_argument("--sum-scale", type=float, default=1.0)
    ap.add_argument("--pol", type=float, default=0.0, help="Polarización para pesos: wP1=(1+pol)/2, wM1=(1-pol)/2")
    ap.add_argument("-o", "--outdir", default="/nfs/cms/arqolmo/ExamplesFCCFullSim/Binned_histograms/")
    ap.add_argument("--var", default="histo")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    f = ROOT.TFile.Open(args.input)
    if not f or f.IsZombie():
        raise RuntimeError(f"No se pudo abrir {args.input}")

    # por bin
    for b in range(args.nBins):
        binName = f"_{b}"
        h_sig = f.Get(args.var + "_SIGNAL" + binName)
        h_p1  = f.Get(args.var + "_SIGNAL_P1" + binName)
        h_m1  = f.Get(args.var + "_SIGNAL_M1" + binName)
        if not all([h_sig, h_p1, h_m1]):
            raise RuntimeError(f"Histos faltantes en bin {b}")
        draw_one(f"c_bin_{b}", args.outdir, f"bin_{b}", h_sig, h_p1, h_m1, args.sum_scale, args.pol)

    # full
    h_sig = f.Get(args.var + "_SIGNAL_full")
    h_p1  = f.Get(args.var + "_SIGNAL_P1_full")
    h_m1  = f.Get(args.var + "_SIGNAL_M1_full")
    if not all([h_sig, h_p1, h_m1]):
        raise RuntimeError("Histos faltantes en full")
    draw_one("c_full", args.outdir, "full", h_sig, h_p1, h_m1, args.sum_scale, args.pol)

    f.Close()

if __name__ == "__main__":
    main()