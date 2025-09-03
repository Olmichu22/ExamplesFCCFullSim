# ComparePlots.py
import ROOT
import argparse
import yaml
import os
from modules.plotting import plot_compare_1D_across_files  # <- función que añadimos abajo

parser = argparse.ArgumentParser(
    description="Comparar histogramas/TGraphs 1D entre varios ROOT files",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter
)
parser.add_argument(
    "-c", "--config", required=True,
    help="Archivo YAML con la configuración de datasets y plots"
)
parser.add_argument(
    "-o", "--outdir", default="MLPF_PANDORAPFO_COMPARE_PLOTS",
    help="Carpeta de salida para las imágenes"
)
args = parser.parse_args()

with open(args.config, "r") as f:
    cfg = yaml.safe_load(f)

ROOT.gStyle.SetOptStat(0)

# --- Cargar ROOT files declarados en el YAML
files_info = []
for ds in cfg.get("datasets", []):
    # cada 'ds' debe tener: path (rootfile), label, (opcional) color, linestyle, markerstyle
    rf = ROOT.TFile(ds["path"])
    if not rf or rf.IsZombie():
        raise RuntimeError(f"No se pudo abrir ROOT file: {ds['path']}")
    files_info.append({
        "file": rf,
        "label": ds.get("label", os.path.basename(ds["path"])),
        "color": ds.get("color", 1),           # ROOT.kBlack por defecto
        "linestyle": ds.get("linestyle", 1),   # sólida
        "markerstyle": ds.get("markerstyle", 20),
        "markersize": ds.get("markersize", 1.5)
    })

# --- Salida
outdir = args.outdir
os.makedirs(outdir, exist_ok=True)

# --- Comparar 1D/TGraph (múltiples figuras)
plot_compare_1D_across_files(
    files_info=files_info,
    plots=cfg.get("plots", {}),
    outdir=outdir
)

print(f"Listo. Imágenes en: {outdir}")
