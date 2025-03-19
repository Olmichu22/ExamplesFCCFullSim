import ROOT 
from ROOT import TH1F 
import argparse
import yaml
import pandas as pd
import os
import warnings

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
def plot_1D_hist(file, variabs, labels, outputpath):
    """
    Plots individual 1D histograms.
    Each histogram is retrieved from the ROOT file, formatted using the provided labels,
    and then saved as a PNG file in the "1D" subfolder of outputpath.
    """
    # Create output folder for 1D plots
    out_dir = os.path.join(outputpath, "1D")
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    
    # Create a canvas for drawing
    c = ROOT.TCanvas("c_1D", "1D Histograms", 800, 600)
    # Loop over each variable
    for var in variabs:
        histo = file.Get(var)
        if not histo:
            print(f"Warning: Histogram '{var}' not found in the ROOT file.")
            continue
        # If label config exists, set axis titles and overall title
        if var in labels:
            cfg = labels[var]
            histo.SetXTitle(cfg.get("x", ""))
            histo.SetYTitle(cfg.get("y", ""))
            histo.SetTitle(cfg.get("title", ""))
        c.Clear()
        histo.Draw()
        out_file = os.path.join(out_dir, f"{var}.png")
        c.SaveAs(out_file)
        print(f"Saved histogram '{var}' as '{out_file}'")
    c.Close()


def plot_hist_together(file, together_config, outputpath):
    """
    Plots groups of 1D histograms together on a single canvas.
    For each group defined in the plot_together configuration, all histograms are drawn
    on the same canvas with different colors and a legend.
    The resulting canvas is saved in a "together" subfolder inside the "1D" folder.
    """
    # Create output folder for together plots under 1D/together
    out_dir = os.path.join(outputpath, "1D", "together")
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    
    # Iterate over each group in the together configuration
    for group, cfg in together_config.items():
        c = ROOT.TCanvas(f"c_together_{group}", group, 800, 600)
        legend = ROOT.TLegend(0.65, 0.70, 0.90, 0.90)
        legend.SetBorderSize(0)
        first = True
        
        # For each histogram name in the group configuration
        for i, var in enumerate(cfg["variabs"]):
            histo = file.Get(var)
            if not histo:
                print(f"Warning: Histogram '{var}' not found in the ROOT file.")
                continue
            # Set the common X and Y axis titles from the group config
            histo.SetXTitle(cfg.get("x", ""))
            histo.SetYTitle(cfg.get("y", ""))
            # Optionally, set the title of the histogram (or leave it empty)
            histo.SetTitle(cfg.get("title", ""))
            
            # Assign a distinct line color for each histogram (simple scheme)
            if i == 0:
                histo.SetLineColor(ROOT.kRed)
            elif i == 1:
                histo.SetLineColor(ROOT.kBlue)
            elif i == 2:
                histo.SetLineColor(ROOT.kGreen+2)
            else:
                histo.SetLineColor(ROOT.kMagenta)
            histo.SetLineWidth(2)
            
            # Draw the first histogram normally; then draw others with "same"
            if first:
                histo.Draw()
                first = False
            else:
                histo.Draw("same")
            # Add legend entry using the corresponding label from config, if provided
            label = cfg["labels"][i] if i < len(cfg["labels"]) else var
            legend.AddEntry(histo, label, "l")
        
        legend.Draw()
        out_file = os.path.join(out_dir, f"{group}.png")
        c.SaveAs(out_file)
        print(f"Saved group '{group}' as '{out_file}'")
        c.Close()


def plot_2D_hist(file, variabs, labels, outputpath):
    """
    Plots 2D histograms.
    Each 2D histogram is drawn with the "COLZ" option and saved as a PNG file
    in the "2D" subfolder of outputpath.
    """
    # Create output folder for 2D plots
    out_dir = os.path.join(outputpath, "2D")
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    
    c = ROOT.TCanvas("c_2D", "2D Histograms", 800, 600)
    for var in variabs:
        histo = file.Get(var)
        if not histo:
            print(f"Warning: 2D Histogram '{var}' not found in the ROOT file.")
            continue
        if var in labels:
            cfg = labels[var]
            histo.SetXTitle(cfg.get("x", ""))
            histo.SetYTitle(cfg.get("y", ""))
            histo.SetTitle(cfg.get("title", ""))
        c.Clear()
        histo.Draw("COLZ")
        out_file = os.path.join(out_dir, f"{var}.png")
        c.SaveAs(out_file)
        print(f"Saved 2D histogram '{var}' as '{out_file}'")
    c.Close()

def plot_cm(results_df, outputpath):
    """
    Generates and saves two confusion matrix plots:
      1. With absolute values.
      2. With normalized values (per actual class) expressed as percentages.
    
    Uses sklearn to compute the confusion matrix and matplotlib to plot it.
    Assumes that 'results_df' contains two columns: 'true' (actual labels) and 'pred' (predicted labels).
    Each cell in the plot is annotated with its corresponding value (number or percentage).
    """

    # Extract true and predicted labels
    y_true = results_df['True']
    y_pred = results_df['Predicted']

    # Get the sorted list of unique classes present
    classes = np.unique(np.concatenate((y_true.values, y_pred.values)))
    
    # Compute the confusion matrix using sklearn
    cm = confusion_matrix(y_true, y_pred, labels=classes)
    
    # Create output directory for the confusion matrix if it does not exist
    cm_dir = os.path.join(outputpath, "CM")
    if not os.path.exists(cm_dir):
        os.makedirs(cm_dir)
    
    # --- Plot with absolute values ---
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title("Confusion Matrix (Absolute Values)")
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    
    thresh = cm.max() / 2.
    # Annotate each cell with the absolute value
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], 'd'),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
    
    plt.ylabel('Actual Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(os.path.join(cm_dir, "confusion_matrix_absolute.png"))
    plt.close()
    
    # --- Normalized plot (per column) ---
    # Normalize the matrix: divide each column by the total number of predicted class elements
    cm_normalized = cm.astype('float') / cm.sum(axis=0, where=cm.sum(axis=0) != 0)[np.newaxis, :]
    cm_normalized = np.nan_to_num(cm_normalized)  # Replace NaN with 0 for cases where the sum is 0
    plt.figure(figsize=(8, 6))
    plt.imshow(cm_normalized, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title("Confusion Matrix (Normalized)")
    plt.colorbar()
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    
    thresh_norm = cm_normalized.max() / 2.
    # Annotate each cell with the percentage (%)
    for i in range(cm_normalized.shape[0]):
        for j in range(cm_normalized.shape[1]):
            percentage = cm_normalized[i, j] * 100
            plt.text(j, i, f"{percentage:.1f}%",
                     horizontalalignment="center",
                     color="white" if cm_normalized[i, j] > thresh_norm else "black")
    
    plt.ylabel('Actual Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(os.path.join(cm_dir, "confusion_matrix_normalized.png"))
    plt.close()
    print(f"Saved confusion matrices to '{cm_dir}'")


parser = argparse.ArgumentParser(description="Configure the plot",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("-i", "--input", default="Results/TauReco/effis0.4_tph0.1_tpi0.1_n1", help="Input path where data to process is stored")
parser.add_argument("-d","--decay", default=-777, type=str) #,"decay0","decay1","decay10"]
parser.add_argument("-p","--plotconfig",default="config/plots/taulong_plotconfig.yaml", type=str)
parser.add_argument("-t", "--typeplot", default="All", type=str)
parser.add_argument("-s", "--same", default="True", type=str, help="If True, Reco, Gen and MatchedGen are plotted in the same plot")

type_plot = ["All", "1D", "2D", "CM"]

args = parser.parse_args()
inputBasePath = args.input
decay = args.decay
plot_config_path = args.plotconfig
typeplot = args.typeplot

if typeplot not in type_plot:
  raise Exception(f"Type of plot not valid. Choose between {type_plot}")

# Load Plot Config File
with open(plot_config_path, "r") as file:
  plot_config = yaml.safe_load(file)

# Load config file of the analysis
with open(inputBasePath+"/config.yaml", "r") as yaml_file:
  config = yaml.safe_load(yaml_file)

# print(config)
# Do not show statistics
ROOT.gStyle.SetOptStat(0)


decay_str = "decayAll" if decay == -777 else f"decay{decay}"

outputpath = inputBasePath+"/Images_"+decay_str

if not os.path.exists(outputpath):
    os.makedirs(outputpath)

input_file = None
labels_file = None
# Look for the output file in the outputfile list
for outputfile in config["output"]["outputfile"]:
  if decay_str in outputfile:
    input_file = str(outputfile)  # Ensure file is a string
    break
for outputfile in config["output"]["outputlabels"]:
  if decay_str in outputfile:
    labels_file = str(outputfile)  # Ensure labels_file is a string
    break

if input_file == None:
  raise Exception(f"Output file not found for decay {decay_str}")
if labels_file == None:
  # Warning if labels file is not found
  warnings.warn(f"Labels file not found for decay {decay_str}. CM plot will not be generated.")
  

file_path = config["output"]["outputpath"]+input_file
true_predict_labels = labels_file

# Open the file
file = ROOT.TFile(file_path)

def formatHisto(file,variab,rename,titleX,color=ROOT.kBlack, linestyle=1, fillstyle=3001):
  histo = file.Get(variab)
  histo.SetName(rename)
  if titleX != None:
    histo.SetXTitle(titleX)
  histo.SetLineColor(color)
  # histo.SetLineWidth(2)
  histo.SetLineStyle(linestyle)
  if fillstyle != None:
    histo.SetFillStyle(fillstyle)
    histo.SetFillColor(color)
  #histo.SetMarkerColor(color)
  #histo.SetMarkerStyle(20)
  # histo.Sumw2()
  return histo

# Select case for the type of plot
if typeplot == "1D":
  variabs = plot_config["variabs_hist"]
  labels = plot_config["plot_titles_config_hist"]
  plot_1D_hist(file, variabs, labels, outputpath)
  if args.same == "True":
    variabs_and_config = plot_config["plot_together"]
    plot_hist_together(file, variabs_and_config, outputpath)
  # Additional logic for 1D plots can be added here
elif typeplot == "2D":
  variabs = plot_config["variabs_2d"]
  labels = plot_config["plot_titles_config_2d"]
  plot_2D_hist(file, variabs, labels, outputpath)
elif typeplot == "CM" and labels_file != None:
  results_df = pd.read_csv(true_predict_labels)
  plot_cm(results_df, outputpath)
elif typeplot == "All":
  variabs = plot_config["variabs_hist"]
  labels = plot_config["plot_titles_config_hist"]
  plot_1D_hist(file, variabs, labels, outputpath)
  variabs = plot_config["variabs_2d"]
  labels = plot_config["plot_titles_config_2d"]
  plot_2D_hist(file, variabs, labels, outputpath)
  results_df = pd.read_csv(true_predict_labels)
  if labels_file != None:
    plot_cm(results_df, outputpath)