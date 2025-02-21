import sys, os, math 
from array import array
import ROOT
from ROOT import TFile, TTree, TH1F, TH2F
import numpy as np
from podio import root_io
import edm4hep
from pathlib import Path

from modules import ZReco
from modules import myutils

import argparse

parser = argparse.ArgumentParser(description="Configure the test",
                         formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("-t", "--type", default="complete", help="Complete (look for every Z) or single (look for a single Z)")

test_type = parser.parse_args().type

sample = "ZTauTau_SMPol_25Sept_MuonFix"
path="/pnfs/ciemat.es/data/cms/store/user/cepeda/FCC/FullSim/"
file="out_reco_edm4hep_edm4hep"
dir_path=path+"/"+sample

nfiles = 5
filenames=[]
for i in range(nfiles):
    filename=dir_path+"/"+file+"_{}.root".format(i)
    print(filename)
    my_file = Path(filename)
    if my_file.is_file():
        root_file = myutils.open_root_file(filename)
        if not root_file or root_file.IsZombie():
            continue
        filenames.append(filename)

if len(filenames)==0:
    print ("No files found")
    sys.exit()
print ("Read %d files" %len(filenames))
reader = root_io.Reader(filenames)
genparts = "MCParticles"
pfobjects="PandoraPFOs"

countEvents=0
for event in reader.get("events"):

   if countEvents%500==0:
      print ("... %d" %countEvents)
   countEvents+=1
   mc_particles = event.get(genparts)
   pfos = event.get(pfobjects)
   
   if test_type=="complete":
      genZs = ZReco.findAllGenZs(mc_particles)
   elif test_type=="single":
      genZ, ZFound = ZReco.findOneZ(mc_particles)
      if ZFound:
        sys.exit()
   else:
      print ("Unknown test type")
      sys.exit()