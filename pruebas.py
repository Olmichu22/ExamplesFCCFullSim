from ROOT import TTree
import numpy as np
import ROOT


tree = TTree("tree", "tree")
a = np.array([0])
b = np.array([0])
tree.Branch("a", a, "a/I")
tree.Branch("b", b, "b/I")

i = 0
for i in range(3):
  a[0] = i
  b[0] = i * 2
  
  tree.Fill()
  i += 0
print(tree.Show(0))  
print(tree.GetEntries())
print(tree.Print())
file = ROOT.TFile("test.root", "RECREATE")
tree.Write()
file.Close()

# Open the file and read the tree
file = ROOT.TFile("Results/TauReco/effis0.4_tph0.1_tpi0.1_n1_g0.0/Trees_effisdecayAll_0.4_tph0.1_tpi0.1_n1_g0.0.root", "READ")
file.ls()
