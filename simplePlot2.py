import ROOT 
from ROOT import TH1F 
import argparse

parser = argparse.ArgumentParser(description="Configure the plot",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("-g","--tag",default="0.4_0.1_1")
parser.add_argument("-f","--sample",default="decayAll", type=str) #,"decay0","decay1","decay10"]
# parser.add_argument("-l","--labels",default=["all taus"], nargs='+', type=str) #, "#pi", "#rho","a_{1} (3#pi)"]
parser.add_argument("-v","--variab",default="TauP", type=str) 
parser.add_argument("-c","--color",default="kBlack", type=str) # ,ROOT.kRed,ROOT.kBlue,ROOT.kGreen+2]
parser.add_argument("-x","--xLabel",default="None", type=str) # ,ROOT.kRed,ROOT.kBlue,ROOT.kGreen+2]
parser.add_argument("-l", "--logscale", default=["0","0"], nargs='+', type=str) # x axis, y axis, 0 means no logscale
parser.add_argument("-o", "--onlyvis", default="True", type=str, help="Only visible taus values")
parser.add_argument("-r", "--xrange", default=None, nargs='+', type=float, help="Range of x axis")
# Do not show statistics
ROOT.gStyle.SetOptStat(0)

args = parser.parse_args()
tag = args.tag
sample = args.sample
# labels = args.labels
variab = args.variab
xLabel = args.xLabel
color = getattr(ROOT,args.color)
onlyvis = True if args.onlyvis == "True" else False
logscale = [int(i) for i in args.logscale]
if args.xrange != None:
  xrange = [float(i) for i in args.xrange]

if xLabel == "None":
  if onlyvis:
    xLabel = variab[:3]+" Vis "+variab[3:] + " (GeV)"
  else:
    xLabel = variab[:3]+" "+variab[3:]

# Select the variables
base_variab_name = "histo"
types = ["Reco", "Gen", "MatchedGen"]

if onlyvis:
  variabs = []
  for type_i in types:
    if type_i == "Reco":
      variabs.append(base_variab_name+type_i+variab)
    else:
      variabs.append(base_variab_name+type_i+variab[:3]+"Vis"+variab[3:])
else:
  variabs = [base_variab_name+type_i+variab for type_i in types]

print("Histograms loaded:\n", variabs)

labels_dict = {"Gen":"Gen", "Matched":"Gen Matched", "Reco":"Reco"}
LineStyle_dict = {"Gen":5, "Matched":1, "Reco":1}
FillStyle_dict = {"Gen":None, "Matched":None, "Reco":3001}

# Open the file
file = ROOT.TFile("effis_"+sample+"_"+tag+".root")

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

# One canvas for all variables
c = ROOT.TCanvas("c")
leg = ROOT.TLegend(0.75,0.89,0.95,0.75)
leg.SetFillStyle(0)
leg.SetFillColor(0)
leg.SetLineColor(0)

histo = {}
for i in range(0, len(variabs)):
  # Get the label
  if "Matched" in variabs[i]:
    label = "Matched"
  elif "Gen" in variabs[i]:
    label = "Gen"
  elif "Reco" in variabs[i]:
    label = "Reco"
  else:
    raise Exception("Label not found")
  
  # xLabel only if it is the first variable
  if i == 0:
    xLabel = xLabel
  else:
    xLabel = None
    
  histo[i] = formatHisto(file, 
                         variabs[i],
                         labels_dict[label],
                         xLabel,
                         color,
                         linestyle=LineStyle_dict[label],
                         fillstyle=FillStyle_dict[label])
  # Set the legend
  legstyle = "l" if label != "Reco" else "lf"
  # print(legstyle)
  leg.AddEntry(histo[i],labels_dict[label],legstyle)

# Check for maximum values
histo[0].Draw()
max_val_y = histo[0].GetMaximum()
# max_val_x = histo[0].GetBinCenter(histo[0].GetMaximumBin())
for i in range(1,len(variabs)):
  histo[i].Draw("same")
  if histo[i].GetMaximum() > max_val_y:
    max_val_y = histo[i].GetMaximum()
  # print(histo[i].GetBinCenter(histo[i].GetMaximumBin()))
  # if histo[i].GetBinCenter(histo[i].GetMaximumBin()) > max_val_x:
  #   max_val_x = histo[i].GetBinCenter(histo[i].GetMaximumBin())

# Set the maximum values
histo[0].SetMaximum(max_val_y*1.4)
histo[0].GetYaxis().SetTitle("Events (not normalized to xsec)")
if xrange != None:
  histo[0].GetXaxis().SetRangeUser(xrange[0], xrange[1])
  

# Set the logscale
if logscale[0] == 1:
  c.SetLogx()
if logscale[1] == 1:
  c.SetLogy()

leg.Draw()
c.SaveAs("effis_"+sample+"_"+variab+"_"+tag+".png")