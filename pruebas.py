from ROOT import TTree
import numpy as np
import ROOT

tree = TTree("tree", "tree")
a = np.array([0])
b = np.array([0])
c = ROOT.std.vector('int')()
tree.Branch("a", a, "a/I")
tree.Branch("b", b, "b/I")
tree.Branch("c", c)

for i in range(4):
  c.clear()  # Vacía el vector antes de cada iteración
  a[0] = i
  b[0] = i * 2
  if i == 0:
    c.push_back(1)
    c.push_back(2)
    c.push_back(3)
  elif i == 1:
    c.push_back(4)
    c.push_back(5)
  elif i == 2:
    c.push_back(6)
  tree.Fill()

print(tree.Show(0))  
print(tree.GetEntries())
print(tree.Print())

for u, entry in enumerate(tree):
  print("Entry", u)
  print(entry.a, entry.b)
  for j in range(entry.c.size()):
    print(entry.c[j])

file = ROOT.TFile("test.root", "RECREATE")
tree.Write()
file.Close()

# Abre el archivo y lee el árbol
file = ROOT.TFile("Results/TauReco/effis0.4_tph0.1_tpi0.1_n1_g0.0/Trees_effisdecayAll_0.4_tph0.1_tpi0.1_n1_g0.0.root", "READ")
file.ls()
