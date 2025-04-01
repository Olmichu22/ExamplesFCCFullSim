import ROOT 
from ROOT import TGraph
import argparse
import yaml
import pandas as pd
import os
from modules import myutils
import warnings
from modules.plotting import plot_1D_hist, plot_2D_hist, plot_hist_together, plot_cm
import numpy as np
import pprint
from array import array
from sklearn.metrics import confusion_matrix
parser = argparse.ArgumentParser(description="Configure the plot",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("-i", "--input", default="Results/Experiments/exps_NeutronCut0.1_TauPhotonPCutExp_TauPionPCut0.1_dRMax0.4_MatchedGenMinDR1.0_generalPCut0.0", help="Input path where data to process is stored")
# parser.add_argument("-d","--decay", default = 10, type=int, help="Decay to plot as principal.") #,"decay0","decay1","decay10"]
parser.add_argument("-p","--plotconfig",default="config/plots/taulong_plotconfig.yaml", type=str)
parser.add_argument(
    "-m", "--migrations", 
    nargs="+",  # Permite múltiples valores separados por espacios
    default = [0, 1, 2],  # Valor por defecto
    type=int, 
    help="List of migrations to plot. Example: 0 1 2."
)
parser.add_argument(
    "-s",
    "--same",
    default="True",
    type=str,
    help="If True, data is plotted in the same plot",
)


args = parser.parse_args()
inputBasePath = args.input
# decay = args.decay
plot_config_path = args.plotconfig
migrations = args.migrations

# default_config = "config/plots/experiment_plotconfig.yaml"
plot_config = args.plotconfig
if plot_config:
  with open(plot_config_path, "r") as yaml_file:
    plot_config = yaml.safe_load(yaml_file)


# Load config file of the analysis
with open(inputBasePath+"/config.yaml", "r") as yaml_file:
  config = yaml.safe_load(yaml_file)

# print(config)
# Do not show statistics
ROOT.gStyle.SetOptStat(0)

outputpath = inputBasePath+"/Images/"

if not os.path.exists(outputpath):
    os.makedirs(outputpath)



# get experiments values
experiment_key = config["experiment"]
experiment_values = config["cuts"][experiment_key]
experiment_values.sort()

values_to_plot = dict()

experiments_files = {}
# Sort experiments files by ascending order regarding experiment_values
for exp_value in experiment_values:
  for exp_file in config["output"]["outputlabels"]:
    if experiment_key+"_"+str(exp_value) in exp_file:
      experiments_files[exp_value] = pd.read_csv(exp_file)
      break

if plot_config:
  migrations = plot_config["migrations"]
  for key in list(plot_config["migrations"].keys()):
    values_to_plot[key] = []
    values_to_plot[key + " norm"] = []
  max_absolute_value = 0
  min_absolute_value = 0
  for exp in experiment_values:
    data = experiments_files[exp]
    true_label = data["True"].to_numpy()
    pred_label = data["Predicted"].to_numpy()
    # Mostrar cm con las labels
    for key, item in plot_config["migrations"].items():
      decay = item[0]
      m = item[1]
      mig_value = np.sum((true_label == decay)*(pred_label == m))
      values_to_plot[key].append(np.sum(mig_value))
      values_to_plot[key + " norm"].append(mig_value/np.sum(true_label == decay))
      if mig_value > max_absolute_value:
        max_absolute_value = mig_value
      if mig_value < min_absolute_value:
        min_absolute_value = mig_value
else:
  # We get all the migrations between m
  for i in range(len(migrations)):
    for j in range(len(migrations)):
      decay = migrations[i]
      m = migrations[j]
      if decay <= -20:
        continue
      values_to_plot[str(decay) + "->" + str(m)] = []
      values_to_plot[str(decay) + "->" + str(m) + " norm"] = []

  max_absolute_value = 0
  min_absolute_value = 0
  for exp in experiment_values:
    data = experiments_files[exp]
    true_label = data["True"].to_numpy()
    pred_label = data["Predicted"].to_numpy()
    # Mostrar cm con las labels
    for m_i in range(len(migrations)):
      decay = migrations[m_i]
      for m_j in range(len(migrations)):
        m = migrations[m_j]
        mig_value = np.sum((true_label == decay)*(pred_label == m))
        if decay<=-20:
          continue
        values_to_plot[str(decay) + "->" + str(m)].append(np.sum(mig_value))
        values_to_plot[str(decay) + "->" + str(m) + " norm"].append(mig_value/np.sum(true_label == decay))
        if mig_value > max_absolute_value:
          max_absolute_value = mig_value
        if mig_value < min_absolute_value:
          min_absolute_value = mig_value



# pprint.pprint(values_to_plot)
# exit()
# 
graphs = {}
for key, values in values_to_plot.items():
  # print(len(experiment_values))
  # print(experiment_values)
  x = array('d', experiment_values)
  y = array('d', values)
  graphs[key] = TGraph(len(experiment_values), x, y)
# exit()

# Create a canvas
canvas_absolute = ROOT.TCanvas("canvas", f'Experiment of {config["experiment"]}', 800, 600)
canvas_normalized = ROOT.TCanvas("canvas_normalized", f'Experiment of {config["experiment"]}', 800, 600)
# Define a color palette
colors = [ROOT.kRed, ROOT.kBlue, ROOT.kGreen, ROOT.kMagenta, ROOT.kCyan, ROOT.kOrange, ROOT.kYellow, ROOT.kBlack, ROOT.kViolet]

# Create a legend
legend_absolute = ROOT.TLegend(0.7, 0.7, 0.9, 0.9)
legend_absolute.SetBorderSize(0)
legend_absolute.SetFillStyle(0)
legend_absolute.SetHeader("Migrations")

# legend norm
legend_normalized = ROOT.TLegend(0.7, 0.7, 0.9, 0.9)
legend_normalized.SetBorderSize(0)
legend_normalized.SetFillStyle(0)
legend_normalized.SetHeader("Migrations")


normalized_keys = [key for key in values_to_plot.keys() if "norm" in key]
absolute_keys = [key for key in values_to_plot.keys() if "norm" not in key]


ncols_legend = 2
legend_absolute.SetNColumns(ncols_legend)
legend_normalized.SetNColumns(ncols_legend)

title = plot_config.get("axis",{}).get("title",f'Evolution of migration by {config["experiment"]}')
xaxis = plot_config.get("axis",{}).get("xlabel",config["experiment"] + "(GeV)")

if plot_config:
  migrations = set()
  for key in plot_config["migrations"].keys():
    decay = plot_config["migrations"][key][0]
    m = plot_config["migrations"][key][1]
    migrations.add(decay)
    migrations.add(m)
  migrations = list(migrations)
    

mig_str = ""
for i in range(len(migrations)):
  mig_str += str(migrations[i]) +"_"
mig_str = mig_str[:-1]

canvas_absolute.cd()


for i, key in enumerate(absolute_keys):
  color = colors[i % len(colors)]
  graph = graphs[key]
  graph.SetLineColor(color)
  graph.SetMarkerColor(color)
  graph.SetMarkerStyle(20)
  graph.SetLineWidth(2)

  if i == 0:
    
    graph.SetTitle(title)
    graph.GetXaxis().SetTitle(xaxis)
    graph.GetYaxis().SetTitle("Counts")
    graph.GetYaxis().SetRangeUser(0.9*min_absolute_value, 1.1*max_absolute_value)
    graph.Draw("alp")  # Draw axis, line, and points for the first graph
  else:
    graph.Draw("lp")  # Draw line and points for subsequent graphs
  legend_absolute.AddEntry(graph, key, "lp")

# Draw the legend
legend_absolute.Draw()

# Save the canvas as an image
canvas_absolute.SaveAs(outputpath + f"graphs_plot_{mig_str}.png")

canvas_normalized.cd()
for i, key in enumerate(normalized_keys):
  color = colors[i % len(colors)]
  graph = graphs[key]
  graph.SetLineColor(color)
  graph.SetMarkerColor(color)
  graph.SetMarkerStyle(20)
  graph.SetLineWidth(2)
  
  if i == 0:
    graph.SetTitle(title)
    graph.GetXaxis().SetTitle(xaxis)
    graph.GetYaxis().SetTitle("Normalized Counts (per True)")
    graph.GetYaxis().SetRangeUser(0, 1.2)
    
    graph.Draw("alp")  # Draw axis, line, and points for the first graph
  else:
    # print("Dibujando segundo")
    graph.Draw("lp")  # Draw line and points for subsequent graphs
  legend_normalized.AddEntry(graph, key, "lp")

# Draw the legend
legend_normalized.Draw()

# Save the canvas as an image
canvas_normalized.SaveAs(outputpath + f"graphs_plot_normalized_{mig_str}.png")

