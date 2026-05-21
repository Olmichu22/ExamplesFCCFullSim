import sys, os, math 
from array import array
import ROOT
from ROOT import TFile, TTree, TH1F, TH2F
import numpy as np
from podio import root_io
import edm4hep
from pathlib import Path
from modules import myutils, tauReco
 
# Load config (necessary for set up the logger)
default_config = "config/default/taurecolong.yaml"
# Output Configuration
outputbasepath = "Results/TauReco/"

general_configs = myutils.setup_analysis_config(default_config, outputbasepath)


loggers = general_configs["loggers"]

run_config = general_configs["config"]

# config = myutils.load_yaml_config(args.config, default_config)


# Cut Configuration
dRMax=run_config["cuts"]["dRMax"]
minPTauPhoton =run_config["cuts"]["TauPhotonPCut"]
minPTauPion = run_config["cuts"]["TauPionPCut"]
PNeutron = run_config["cuts"]["NeutronCut"]
dRMatch = run_config["cuts"]["MatchedGenMinDR"]
generalPCut = run_config["cuts"]["generalPCut"]

selectDecay=general_configs["decay"]

outputpath = general_configs["outputpath"]
fileOutName = os.path.join(general_configs["outputpath"], general_configs["fileOutName"])

logger_config = loggers["config"]
logger_io = loggers["io"]
logger_process = loggers["processing"]
logger_pi0mass = loggers["pi0mass"]


# Continue with the rest of configs

# ------------------------------------------------------------------------
# General Configuration
sample=run_config["general"]["sample"]
matched_cm_arg = general_configs["flags"]["matched_cm"]
test_arg = general_configs["flags"]["test"]

logger_config.info("Configuration loaded!")
# logger_config.info("Configuration:\n%s", pprint.pformat(general_configs, indent=4))


# ------------------------------------------------------------------------
gatr_results_path = general_configs["args"].gatr_result

filenames, mlpf_results = myutils.get_root_trees_path(sample, gatr_results_path, loggers, test_arg)
reader = root_io.Reader(filenames)

fileOutName="particleflowcheck_fiducialcut.root"

# sample="ZTauTau_SMPol_17Sept_1/"
# POL="SMFirst"
# path="/pnfs/ciemat.es/data/cms/store/user/cepeda/FCC/FullSim/"
# file="out_reco_edm4hep_edm4hep"
# filenames=[]
# dir_path=path+"/"+sample
# names = ROOT.std.vector('string')()
# nfiles=len(os.listdir(dir_path))

# nfiles=200
# badfiles=[-1] #11,15]

# print (dir_path)
# for i in range(1,nfiles+1):
#     if (i in badfiles): continue # this one is broken? why?
#     filename=dir_path+"/"+file+"_{}.root".format(i)
#     print (filename)

#     my_file = Path(filename)
#     if my_file.is_file():
#         root_file = tauReco.open_root_file(filename)
#         if not root_file or root_file.IsZombie():
#             continue
#         filenames.append(filename)

# print ("Read %d files" %len(filenames))
# reader = root_io.Reader(filenames)


# collections to use 
genparts = "MCParticles"
pfobjects="PandoraPFOs"

hGenPionsN=TH1F("hGenPionsN","",10,0,10)
hPFPionsN=TH1F("hPFPionsN","",10,0,10)
hPFNeutronsN=TH1F("hPFNeutronsN","",10,0,10)
hGenNeutronsN=TH1F("hGenNeutronsN","",10,0,10)

hPFPionP=TH1F("histoPFPionP","",250,0,50)
hPFNeutronP=TH1F("histoPFNeutronP","",250,0,50)
hGenNeutronP=TH1F("histoGenNeutronsP","",250,0,50)

hGenPionP=TH1F("histoGenPionP","",250,0,50)
hGenPionTheta=TH1F("histoGenPionTheta","",100,0,3.15)
hGenPionPhi=TH1F("histoGenPionPhi","",100,-3.15,3.15)
hGenPionCluster=TH1F("histoGenPionCluster","",5,0,5)

hBestMatchP=TH1F("histoBestMatchP","",250,0,50)
hBestMatchPDG=TH1F("histoBestMatchPDG","",5000,-2500,2500)
hBestMatchTheta=TH1F("histoBestMatchTheta","",100,0,3.15)
hBestMatchPhi=TH1F("histoBestMatchPhi","",100,-3.15,3.15)
hBestMatchDR=TH1F("histoBestMatchDR","Angle Best Match",200,0,0.4)

hMGenPionP=TH1F("histoMGenPionP","",250,0,50)
hMGenPionTheta=TH1F("histoMGenPionTheta","",100,0,3.15)
hMGenPionPhi=TH1F("histoMGenPionPhi","",100,-3.15,3.15)
hMGenPionCluster=TH1F("histoMGenPionCluster","",5,0,5)

hMGenPionP_Pion=TH1F("histoMGenPionP_Pion","",250,0,50)
hMGenPionTheta_Pion=TH1F("histoMGenPionTheta_Pion","",100,0,3.15)
hMGenPionPhi_Pion=TH1F("histoMGenPionPhi_Pion","",100,-3.15,3.15)
hMGenPionDR_Pion=TH1F("histoMGenPionDR_Pion","Angle Best Match",200,0,0.4)
hMGenPionCluster_Pion=TH1F("histoMGenPionCluster_Pion","",5,0,5)

hGenPionCosTheta=TH1F("histoGenPionCosTheta","",100,-1,1)
hMGenPionCosTheta=TH1F("histoMGenPionCosTheta","",100,-1,1)
hMGenPionCosTheta_Pion=TH1F("histoMGenPionCosTheta_Pion","",100,-1,1)
hMGenPionCosTheta_Neutron=TH1F("histoMGenPionCosTheta_Neutron","",100,-1,1)
hMGenPionCosTheta_Other=TH1F("histoMGenPionCosTheta_Other","",100,-1,1)
hBestMatchCosTheta=TH1F("histoBestMatchCosTheta","",100,-1,1)

hMGenPionP_Neutron=TH1F("histoMGenPionP_Neutron","",250,0,50)
hMGenPionTheta_Neutron=TH1F("histoMGenPionTheta_Neutron","",100,0,3.15)
hMGenPionPhi_Neutron=TH1F("histoMGenPionPhi_Neutron","",100,-3.15,3.15)
hMGenPionDR_Neutron=TH1F("histoMGenPionDR_Neutron","Angle Best Match",200,0,0.4)
hMGenPionPDG_Neutron=TH1F("hMGenPionPDG_Neutron","",5000,-2500,2500)
hMGenPionCluster_Neutron=TH1F("histoMGenPionCluster_Neutron","",5,0,5)

hMGenPionP_Other=TH1F("histoMGenPionP_Other","",250,0,50)
hMGenPionTheta_Other=TH1F("histoMGenPionTheta_Other","",100,0,3.15)
hMGenPionPhi_Other=TH1F("histoMGenPionPhi_Other","",100,-3.15,3.15)
hMGenPionDR_Other=TH1F("histoMGenPionDR_Other","Angle Best Match",200,0,0.4)
hMGenPionPDG_Other=TH1F("hMGenPionPDG_Other","",5000,-2500,2500)
hMGenPionCluster_Other=TH1F("histoMGenPionCluster_Other","",5,0,5)



# run over all events 
for event_idx, event in enumerate(reader.get("events")):
    logger_io.info("Processing event %d", event_idx)
    mc_particles = event.get( genparts )
    pfos = event.get(pfobjects)

    #print ("-----------")

    countPionsPF=0
    countNeutronsPF=0
    for pf in pfos:
       pfP4=ROOT.TLorentzVector()
       pfP4.SetXYZM(pf.getMomentum().x,pf.getMomentum().y,pf.getMomentum().z,pf.getMass())
       pfPDG=pf.getPDG()
       if (abs(pfPDG)==211):
          countPionsPF+=1
          hPFPionP.Fill(pfP4.P())
       if (pfPDG==2112):
          countNeutronsPF+=1
          hPFNeutronP.Fill(pfP4.P())
     #  print (".... all pf?",pfPDG,pf.getMass(),pfP4.P(),pfP4.Theta(),pfP4.Phi())

    countPionsGEN=0
    countNeutronsGEN=0

    for mc in mc_particles:
       if abs(mc.getPDG())==211:
          countPionsGEN+=1
       if abs(mc.getPDG())==2112:
          countNeutronsGEN+=1

    # Fill Counts
    hGenPionsN.Fill(countPionsGEN)
    hPFPionsN.Fill(countPionsPF)
    hPFNeutronsN.Fill(countNeutronsPF)
    hGenNeutronsN.Fill(countNeutronsGEN)

    for mc in mc_particles:
       mcP4=ROOT.TLorentzVector()
       mcP4.SetXYZM(mc.getMomentum().x,mc.getMomentum().y,mc.getMomentum().z,mc.getMass())    
       mcPDG=abs(mc.getPDG())
       if mcPDG==2112:
         hGenNeutronP.Fill(mcP4.P())

       if abs(mcPDG)!=211:
          continue
       #print("... mc ",mcPDG,mcP4.P(),mcP4.Theta(),mcP4.Phi())

       if abs(math.cos(mcP4.Theta()))>0.95:
          continue

       pionCluster=1
       # does it have other pions around?
       for mc2 in mc_particles:
           if mc2==mc:
              continue 
           mc2P4=ROOT.TLorentzVector()
           mc2P4.SetXYZM(mc2.getMomentum().x,mc2.getMomentum().y,mc2.getMomentum().z,mc2.getMass())
           mc2PDG=abs(mc2.getPDG())
           if abs(mc2PDG)!=211:
                continue
           angleGenPions=myutils.dRAngle(mc2P4,mcP4)
           if angleGenPions<0.1:
              pionCluster+=1


       hGenPionP.Fill(mcP4.P())
       if mcP4.P()>10:
        hGenPionTheta.Fill(mcP4.Theta())
        hGenPionPhi.Fill(mcP4.Phi())
        hGenPionCluster.Fill(pionCluster)
        hGenPionCosTheta.Fill(math.cos(mcP4.Theta()))

       if len(pfos)==0:
           continue

       closestDR=1
       match=pfos[0]       
       # find the closest pfo to this pion
       for pf in pfos: 
         pfP4=ROOT.TLorentzVector()
         pfP4.SetXYZM(pf.getMomentum().x,pf.getMomentum().y,pf.getMomentum().z,pf.getMass())
         pfPDG=pf.getPDG()
         if pfPDG==22:
            continue         
         angle=myutils.dRAngle(pfP4,mcP4)
         if (angle<closestDR):
                match=pf 
                closestDR=angle

       matchP4=ROOT.TLorentzVector()
       matchP4.SetXYZM(match.getMomentum().x,match.getMomentum().y,match.getMomentum().z,match.getMass())
       matchPDG=match.getPDG()
       #print("... match ",matchPDG,matchP4.P(),matchP4.Theta(),matchP4.Phi(),closestDR)
        
       hBestMatchDR.Fill(closestDR)

       if closestDR>0.1:
            continue

       hBestMatchP.Fill(matchP4.P())
       if matchP4.P()>10:
        hBestMatchTheta.Fill(matchP4.Theta())
        hBestMatchPhi.Fill(matchP4.Phi())
        hBestMatchPDG.Fill(matchPDG)
        hBestMatchCosTheta.Fill(math.cos(matchP4.Theta()))

       hMGenPionP.Fill(mcP4.P())
       if mcP4.P()>10:
        hMGenPionTheta.Fill(mcP4.Theta())
        hMGenPionPhi.Fill(mcP4.Phi())
        hMGenPionCluster.Fill(pionCluster)
        hMGenPionCosTheta.Fill(math.cos(mcP4.Theta()))

       if abs(matchPDG)==211:
          hMGenPionP_Pion.Fill(mcP4.P())
          if mcP4.P()>10: 
           hMGenPionTheta_Pion.Fill(mcP4.Theta())
           hMGenPionPhi_Pion.Fill(mcP4.Phi())
           hMGenPionDR_Pion.Fill(closestDR)
           hMGenPionCluster_Pion.Fill(pionCluster)
           hMGenPionCosTheta_Pion.Fill(math.cos(mcP4.Theta()))

       elif matchPDG==2112:
          hMGenPionP_Neutron.Fill(mcP4.P())
          if mcP4.P()>10:
           hMGenPionTheta_Neutron.Fill(mcP4.Theta())
           hMGenPionPhi_Neutron.Fill(mcP4.Phi())
           hMGenPionPDG_Neutron.Fill(matchPDG)
           hMGenPionDR_Neutron.Fill(closestDR)
           hMGenPionCluster_Neutron.Fill(pionCluster)
           hMGenPionCosTheta_Neutron.Fill(math.cos(mcP4.Theta()))

       else:
          hMGenPionP_Other.Fill(mcP4.P())
          if mcP4.P()>10:
           hMGenPionTheta_Other.Fill(mcP4.Theta())
           hMGenPionPhi_Other.Fill(mcP4.Phi())
           hMGenPionPDG_Other.Fill(matchPDG)
           hMGenPionDR_Other.Fill(closestDR)
           hMGenPionCluster_Other.Fill(pionCluster)
           hMGenPionCosTheta_Other.Fill(math.cos(mcP4.Theta()))


hMGenPionP.Sumw2()
hMGenPionP_Pion.Sumw2()
hGenPionP.Sumw2()

eff_PionP_all=hMGenPionP.Clone()
eff_PionP_all.SetName("eff_PionP_all")
eff_PionP_Pion=hMGenPionP_Pion.Clone()
eff_PionP_Pion.SetName("eff_PionP_Pion")
eff_PionP_all.Divide(hGenPionP)
eff_PionP_Pion.Divide(hGenPionP)

outfile=ROOT.TFile(fileOutName,"RECREATE")

hGenPionP.Write()
hGenPionTheta.Write()
hGenPionPhi.Write()
hBestMatchP.Write()
hBestMatchPDG.Write()
hBestMatchTheta.Write()
hBestMatchPhi.Write()
hBestMatchDR.Write()
hMGenPionP.Write()
hMGenPionTheta.Write()
hMGenPionPhi.Write()
hMGenPionP_Pion.Write()
hMGenPionTheta_Pion.Write()
hMGenPionPhi_Pion.Write()
hMGenPionDR_Pion.Write()
hMGenPionCluster_Pion.Write()
hMGenPionCluster.Write()
hGenPionCluster.Write()
hMGenPionP_Neutron.Write()
hMGenPionTheta_Neutron.Write()
hMGenPionPhi_Neutron.Write()
hMGenPionDR_Neutron.Write()
hMGenPionPDG_Neutron.Write()
hMGenPionCluster_Neutron.Write()
hMGenPionP_Other.Write()
hMGenPionTheta_Other.Write()
hMGenPionPhi_Other.Write()
hMGenPionDR_Other.Write()
hMGenPionPDG_Other.Write()
hMGenPionCluster_Other.Write()

eff_PionP_Pion.Write()
eff_PionP_all.Write()

hPFNeutronsN.Write()
hPFPionsN.Write()
hGenPionsN.Write()
hGenNeutronsN.Write()

hPFPionP.Write()
hPFNeutronP.Write()
hGenNeutronP.Write()

hGenPionCosTheta.Write()
hMGenPionCosTheta.Write()
hMGenPionCosTheta_Pion.Write()
hMGenPionCosTheta_Neutron.Write()
hMGenPionCosTheta_Other.Write()
hBestMatchCosTheta.Write()

outfile.Close()

