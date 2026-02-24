#!/usr/bin/env python3
import os
import ROOT
from ROOT import TFile, TCanvas, TLegend

ROOT.gStyle.SetOptStat(0)
ROOT.gROOT.SetBatch(True)
# =====================================================
# CONFIGURACIÓN DE EVENTOS (rellenar por el usuario)
# =====================================================
EVENT_CONFIG = {
    "Ztt": {
        "xsec_pb": 1476.58,
        "lumi_pb": 6972,
        "ngen": 10296000
    },
    "Zqq": {
        "xsec_pb": 30170,
        "lumi_pb": 6972,
        "ngen": 1000000
    },
    "Bhabha": {
        "xsec_pb": 273500,
        "lumi_pb": 6972,
        "ngen": 2396000
    },
}


def compute_weight(event_type, verbose=False):
    cfg = EVENT_CONFIG.get(event_type)
    if cfg is None:
        raise ValueError(f"Event type '{event_type}' not defined")

    weight = cfg["lumi_pb"] * cfg["xsec_pb"] / cfg["ngen"]

    if verbose:
        print(f"[INFO] {event_type}: weight = {weight}")

    return weight


def main(
    signal_file,
    signal_type,
    background_files,  # list of (file, type)
    output_dir=".",
    suffix="",
    var="OmegaCosTheta",
    title="#omega_{#rho}",
    nBins=20,
    rebin=1,
    make_plots=True,
    verbose=False,
    syst="nominal",  # "nominal", "min", "max"
):

    os.makedirs(output_dir, exist_ok=True)

    # -------------------------
    # Abrir archivos
    # -------------------------
    sig_in = TFile.Open(signal_file)
    if not sig_in or sig_in.IsZombie():
        raise RuntimeError(f"No se pudo abrir {signal_file}")

    bg_inputs = []
    for fname, etype in background_files:
        f = TFile.Open(fname)
        if not f or f.IsZombie():
            raise RuntimeError(f"No se pudo abrir {fname}")
        bg_inputs.append((f, etype))

    # -------------------------
    # Pesos
    # -------------------------
    w_signal = compute_weight(signal_type, verbose)
    w_bgs = {etype: compute_weight(etype, verbose) for _, etype in bg_inputs}

    # -------------------------
    # Sufijo para sistemáticos
    # -------------------------
    if syst == "nominal":
        syst_suffix = ""
        syst_tag = ""
    elif syst == "min":
        syst_suffix = "_min"
        syst_tag = "_syst_min"
    elif syst == "max":
        syst_suffix = "_max"
        syst_tag = "_syst_max"
    else:
        raise ValueError(f"syst debe ser 'nominal', 'min' o 'max', no '{syst}'")

    if verbose and syst != "nominal":
        print(f"[INFO] Usando variación sistemática: {syst}")

    # -------------------------
    # Output
    # -------------------------
    base = os.path.basename(signal_file)
    name, ext = os.path.splitext(base)
    outfile_path = os.path.join(
        output_dir, f"BINED_template_{name}{suffix}{syst_tag}{ext}"
    )
    outfile = TFile(outfile_path, "RECREATE")

    if verbose:
        print(f"[INFO] Output → {outfile_path}")

    # -------------------------
    # Config
    # -------------------------
    samples_signal = ["SIGNAL", "SIGNAL_P1", "SIGNAL_M1"]
    sample_names = ["SM", "A_{#tau}=+1", "A_{#tau}=-1", "BG", "BG", "BG"]
    colors = [ROOT.kRed, ROOT.kGreen+4, ROOT.kGreen+4, ROOT.kYellow]

    bin_length = 100 // nBins

    # =====================================================
    # LOOP POR BINS
    # =====================================================
    for b in range(nBins):
        bin_ini = int(b * bin_length + 1)
        bin_end = int((b + 1) * bin_length)

        if verbose:
            print(f"[BIN {b}] {bin_ini} → {bin_end}")

        histos = {}
        max_y = 0

        # -------------------------
        # Señal (Ztt)
        # -------------------------
        for i, s in enumerate(samples_signal):
            histo_name = f"{var}_{s}{syst_suffix}"
            h2 = sig_in.Get(histo_name)
            if not h2:
                raise RuntimeError(f"{histo_name} no encontrado en señal")

            h1 = h2.ProjectionX(f"histo_{s}_{b}", bin_ini, bin_end)
            h1.Rebin(rebin)
            # if "M1" in s or "P1" in s:
            #   h1.Scale(w_signal/2.0)  # templates tienen la mitad de eventos
            # else:
            #   h1.Scale(w_signal)
            h1.Scale(w_signal)
            
            

            h1.SetXTitle(title)
            h1.SetLineWidth(2)
            h1.SetLineColor(colors[i])

            histos[s] = h1
            max_y = max(max_y, h1.GetMaximum())

        # -------------------------
        # BG total = migraciones + fondos puros
        # -------------------------
        # Migraciones (desde señal)
        h_bg_mig = None

        # Fondos alternativos
        h_bg_zqq = None
        h_bg_bhabha = None
        histos_bg = {"Zqq": h_bg_zqq, "Bhabha": h_bg_bhabha}
        # BG desde señal (migraciones)
        histo_name_bg = f"{var}_BG{syst_suffix}"
        h_bg_mig = sig_in.Get(histo_name_bg)
        if not h_bg_mig:
            raise RuntimeError(f"{histo_name_bg} no encontrado en señal")

        h_bg_mig = h_bg_mig.ProjectionX(f"histo_BG_migrations_{b}", bin_ini, bin_end)
        h_bg_mig.Rebin(rebin)
        h_bg_mig.Scale(w_signal)

        # BG desde otros archivos
        for f, etype in bg_inputs:
            histo_name_bg_ext = f"{var}_BG{syst_suffix}"
            histos_bg[etype] = f.Get(histo_name_bg_ext)
            if not histos_bg[etype]:
                raise RuntimeError(f"{histo_name_bg_ext} no encontrado en fondo {etype}")

            histos_bg[etype] = histos_bg[etype].ProjectionX(
                f"histo_BG_{etype}_{b}", bin_ini, bin_end
            )
            histos_bg[etype].Rebin(rebin)
            histos_bg[etype].Scale(w_bgs[etype])
            # h_bg.Add(h1)
        histos_bg.update({"migrations": h_bg_mig})
        fill_colors = [ROOT.kBlue - 7, ROOT.kCyan - 7, ROOT.kMagenta - 7]
        line_colors = [ROOT.kBlue + 2, ROOT.kCyan + 2, ROOT.kMagenta + 2]
        bg_order = ["migrations", "Zqq", "Bhabha"]
        for i, etype in enumerate(bg_order):
            h1 = histos_bg[etype]
            h1.SetFillColor(fill_colors[i])
            h1.SetLineColor(line_colors[i])
            h1.SetLineWidth(2)
            # if i == 0:
            #     h_bg = h1.Clone(f"histo_BG_{b}")
            # else:
            #     h_bg.Add(h1)
        # h_bg.SetFillColor(colors[3])
        # h_bg.SetLineColor(colors[3])
        # h_bg.SetLineWidth(2)
        max_y_bg = 0
        for h in histos_bg.values():
            max_y_bg = max(max_y_bg, h.GetMaximum())
        max_y = max(max_y, max_y_bg)
        histos.update(histos_bg)

        # -------------------------
        # Plots
        # -------------------------
        if make_plots:
            c = TCanvas(f"c_bin_{b}", "", 800, 800)
            leg = TLegend(0.5, 0.89, 0.9, 0.6)
            leg.SetFillStyle(0)
            leg.SetLineColor(0)
            bg_labels = {
                "migrations": "BG (migrations)",
                "Zqq": "Zqq BG",
                "Bhabha": "Bhabha BG"
            }
            draw_order = ["SIGNAL", "SIGNAL_P1", "SIGNAL_M1"] + list(histos_bg.keys())
            first_drawn = True
            for i, key in enumerate(draw_order):
                h = histos[key]
                # Saltar histogramas sin entradas
                if h.GetEntries() == 0 or h.Integral() == 0:
                    continue
                opt = "HIST" if first_drawn else "HIST SAME"
                h.Draw(opt)
                first_drawn = False
                leg.AddEntry(h, bg_labels.get(key, sample_names[i]), "f" if key in histos_bg else "l")

            # Solo ajustar máximo si hay histogramas dibujados
            if not first_drawn:
                histos["SIGNAL"].SetMaximum(1.2 * max_y)
                leg.Draw()
            c.SaveAs(os.path.join(output_dir, f"BIN_{var}_{b}.png"))

        # -------------------------
        # Escritura
        # -------------------------
        outfile.cd()
        h_bg_tot = h_bg_mig.Clone(f"histo_BG_{b}")
        for h in histos_bg.values():
            if h is not h_bg_mig:
                h_bg_tot.Add(h)
        h_bg_tot.Write()
        for h in histos.values():
            h.Write()
        
    # =========================
    # FULL RANGE (igual que antes)
    # =========================
    bin_ini = 1
    bin_end = 100

    # Señal
    for s in samples_signal:
        histo_name = f"{var}_{s}{syst_suffix}"
        h2 = sig_in.Get(histo_name)
        h1 = h2.ProjectionX(f"histo_{s}_full", bin_ini, bin_end)
        h1.Rebin(rebin)
        # if "M1" in s or "P1" in s:
        #     h1.Scale(w_signal/2.0)  # templates tienen la mitad de eventos
        # else:
        #     h1.Scale(w_signal)
        h1.Scale(w_signal)
        h1.Write()

    # BG migraciones
    histo_name_bg = f"{var}_BG{syst_suffix}"
    h_bg_mig = sig_in.Get(histo_name_bg)
    h_bg_mig = h_bg_mig.ProjectionX("histo_BG_migrations_full", bin_ini, bin_end)
    h_bg_mig.Rebin(rebin)
    h_bg_mig.Scale(w_signal)
    h_bg_mig.Write()

    # BG alternativos
    h_bg_tot = h_bg_mig.Clone("histo_BG_full")
    for f, etype in bg_inputs:
        histo_name_bg_ext = f"{var}_BG{syst_suffix}"
        h = f.Get(histo_name_bg_ext).ProjectionX(
            f"histo_BG_{etype}_full", bin_ini, bin_end
        )
        h.Rebin(rebin)
        h.Scale(w_bgs[etype])
        h.Write()
        h_bg_tot.Add(h)

    h_bg_tot.Write()
                

    outfile.Close()
    sig_in.Close()
    for f, _ in bg_inputs:
        f.Close()

    if verbose:
        print("[OK] Binning con múltiples archivos completado")


# =====================================================
# CLI
# =====================================================
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser("Angular binning with signal + multiple backgrounds")

    parser.add_argument("--signal-file", default="Results/RhoAnalysis/FULL_sample_tau_trained0.4_tph0.35_tpi0_n3_g0.0/Histos_dRgt3.06_5.0_MesonPgt0.0_lt45.57_LeptonPgt0.0_lt41.16_tau_traineddecay2_0.4_tph0.35_tpi0_n3_g0.0.root")
    parser.add_argument("--signal-type", default="Ztt")
    parser.add_argument(
        "--background-files",
        nargs="*",
        default=["Results/RhoAnalysis/Zqq_sample_extremes_tau_trained0.4_tph0.35_tpi0_n3_g0.0/Histos_dRgt3.06_5.0_MesonPgt0.0_lt45.57_LeptonPgt0.0_lt41.16_tau_traineddecay2_0.4_tph0.35_tpi0_n3_g0.0.root:Zqq",
                 "Results/RhoAnalysis/bhabha_sample_FULLtau_trained0.4_tph0.35_tpi0_n3_g0.0/Histos_dRgt3.06_5.0_MesonPgt0.0_lt45.57_LeptonPgt0.0_lt41.16_tau_traineddecay2_0.4_tph0.35_tpi0_n3_g0.0.root:Bhabha"],
        help="Formato: file.root:EventType"
    )
    parser.add_argument("--suffix", default="")
    parser.add_argument("-o", "--outdir", default="./Binned_histograms/")
    parser.add_argument("--nBins", type=int, default=20)
    parser.add_argument("--rebin", type=int, default=1)
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument("-v", "--verbose", action="store_true")
    parser.add_argument("--syst", choices=["nominal", "min", "max"], default="nominal",
                        help="Variación sistemática: nominal (sin sufijo), min (_min), max (_max)")

    args = parser.parse_args()

    bg_files = []
    for item in args.background_files:
        fname, etype = item.split(":")
        bg_files.append((fname, etype))

    main(
        signal_file=args.signal_file,
        signal_type=args.signal_type,
        background_files=bg_files,
        output_dir=args.outdir,
        suffix=args.suffix,
        nBins=args.nBins,
        rebin=args.rebin,
        make_plots=not args.no_plots,
        verbose=args.verbose,
        syst=args.syst,
    )
