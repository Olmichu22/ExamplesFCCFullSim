# More complex example, not yet cleaned 
# Looks for MuTau / ETau combinations 
# Then checks the tau polatization using the hadronic tau 

import yaml
import sys, os, math 
from array import array
import ROOT
from ROOT import TFile, TTree, TH1F, TH2F
import numpy as np
from podio import root_io
import edm4hep
from pathlib import Path
import ctypes
from modules.TauDecays import extractTauDecays

import os
import pickle
import pandas as pd
from modules import tauReco 
from modules import weightsPol
from modules import optimalVariabRho
from modules import myutils 
import pprint
import argparse

from modules import (ParticleObjects, electronReco, muonReco, myutils, pi0Reco,
                     tauReco, particleMatch)

  

def write_histograms_recursive(obj):
    """
    Recorre un diccionario anidado y ejecuta `.Write()` en cada objeto tipo ROOT histogram.
    """
    if isinstance(obj, dict):
        for value in obj.values():
            write_histograms_recursive(value)
    else:
        # Si no es diccionario, asumimos que es un histograma de ROOT
        try:
            obj.Write()
        except AttributeError:
            print(f"Objeto {obj} no tiene método .Write(). Ignorado.")

# ----------------------------------------------------------------------------
# Load config (necessary for set up the logger)
default_config = "config/default/taurecolong.yaml"
# Output Configuration
outputbasepath = "Results/RhoAnalysis/"

def my_hook(parser):
    parser.add_argument("--sys-err", type=str, default="config/systematics/err_sys.yml", help="YAML file with systematics errors to apply")
    parser.add_argument("--test-extremes", action="store_true", help="Test the extremes of the photon energy resolution")
    parser.add_argument("--sin-eff", type=float, default=None, help="Effective sin^2 theta_W to use in the weights calculation")
    
general_configs = myutils.setup_analysis_config(default_config, outputbasepath, parser_hook=my_hook)

loggers = general_configs["loggers"]

run_config = general_configs["config"]

# config = myutils.load_yaml_config(args.config, default_config)


args =  general_configs["args"]

if args.sin_eff is not None:
    loggers["processing"].info(f"Using sin2theta_effective = {args.sin_eff} for weights calculation.")
test_pfo = args.test_pfo
# Cut Configuration
dRMax=run_config["cuts"]["dRMax"]
tauPCut = run_config["cuts"]["tauCut"]
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
sys_errors = general_configs["config"].get("systematics_errors", {})
photon_config = sys_errors.get("photon_config", {})
test_extremes = args.test_extremes
# err_had = sys_errors.get("err_hadrons", {})
# err_ele = sys_errors.get("err_electrons", {})
# err_mu = sys_errors.get("err_muons", {})
logger_config.info("Systematics errors: %s", pprint.pformat(sys_errors, indent=4))
# Continue with the rest of configs

# ------------------------------------------------------------------------
# General Configuration
sample=run_config["general"]["sample"]
matched_cm_arg = general_configs["flags"]["matched_cm"]
test_arg = general_configs["flags"]["test"]

logger_config.info("Configuration loaded!")
logger_config.info("Configuration:\n%s", pprint.pformat(general_configs, indent=4))


# ------------------------------------------------------------------------
gatr_results_path = general_configs["args"].gatr_result

filenames, mlpf_results = myutils.get_root_trees_path(sample, gatr_results_path, loggers, test_arg, args)
reader = root_io.Reader(filenames)
logger_io.info("Read %d files", len(filenames))
logger_io.info("First %s files.", filenames[:10]) 

# Configs and reading finished
# ----------------------------------------------------------------------


# collections to use
genparts = "MCParticles"
pfobjects = "PandoraPFOs"
# pfobjects ="TightSelectedPandoraPFOs"

histogram_config = general_configs.get("histograms_config", {})
root_histograms = myutils.set_up_root_histograms(histogram_config)
if test_extremes:
    logger_process.info("Testing extremes is enabled.")

    root_histograms_super = {
        "original": root_histograms,
        "min_err": myutils.clone_histograms_with_suffix(root_histograms, "_min"),
        "max_err": myutils.clone_histograms_with_suffix(root_histograms, "_max")
    }

else:
    root_histograms_super = {"original": root_histograms}
# miss_matched_histograms_p = dict()
# miss_matched_histograms_theta = dict()

treeName="outtree"
variabs=["genTauP","genMesonP","genPionP","genTauE",
         "genTauM","genMesonE","genMesonM","genPionE",
         "genPionM","genTauTheta","genTauPhi","genMesonTheta",
         "genMesonPhi","genPionTheta","genPionPhi","gen_cos_theta",
         "gen_cos_psi","gen_cos_beta","gen_w","weight_P1",
         "weight_M1","genOmega","gen_cos_theta_tau","recoMesonP",
         "recoPionP","recoMesonTheta","recoMesonPhi","recoPionTheta",
         "recoPionPhi","recoMesonE","recoMesonM","recoPionE",
         "recoPionM","cos_theta","cos_psi","cos_beta",
         "omega","cos_theta_rho","genTauID","recoTauID",
         "ZMass","GenZMass","GenZVisMass","beamE",
         "nPhotonsReco", "nPhotonsGen", "isElectron", "lepP", "lepE", "lepTheta", "lepPhi", "lepPDG"]
outfile=ROOT.TFile(fileOutName,"RECREATE")


trees = {}
branches_super = {}
for key in root_histograms_super:
  trees[key] = ROOT.TTree(treeName + f"_{key}", f"processed variables - {key}")
  branches_super[key] = {}
  
# new_tree = ROOT.TTree(treeName,"processed variables")

  variable_variabs = ["reco_photons_E", "reco_photons_theta", "reco_photons_phi", "gen_photons_E", "gen_photons_theta", "gen_photons_phi"]

  # branches = {}

  for var in variabs:
          branches_super[key][var] = ctypes.c_double(0.0)  # Single double variable
          trees[key].Branch(var, ctypes.addressof(branches_super[key][var]), f"{var}/D")
  for var in variable_variabs:
    branches_super[key][var] = ROOT.std.vector('double')()  # Vector of doubles
    trees[key].Branch(var, branches_super[key][var])
# normalize

# I am sure there is a function that does this but I cannot find it
#totalEvents=0
#for e in reader.get("events"):
#    totalEvents+=1

totalEvents=1#296800
lumi=70*1000 # discuss this with Michele. Prob missing a factor 2
xsecZtautau=1476.58 #pb , from https://fcc-physics-events.web.cern.ch/FCCee/delphes/winter2023/idea/ 
weight=1#xsecZtautau/totalEvents 

loggers["processing"].info(f"Events? {totalEvents}, Xsec (pb) : {xsecZtautau}, Weight : {weight}")

totalEvents=0
selectedEvents=0

sumWeightsP1=0
sumWeightsM1=0
sumWeights=0

# run over all events 
for eventid, event in enumerate(reader.get("events")):
  loggers["processing"].debug("Processing event %d", totalEvents)
  if totalEvents%500==0:
      loggers["processing"].info(f"Processed Events: {totalEvents}")
  for tree_key in trees:
    for var in variable_variabs:
      branches_super[tree_key][var].clear()  # Vector of doubles
  
  # Var to avoid double counting selected events
  prevSelectedEvents=selectedEvents
  
  totalEvents+=1
  mc_particles = event.get( genparts )
  beamE=mc_particles[0].getEnergy()

  ## get GEN level info
  if sample in ("ZTauTau_SMPol_25Sept_MuonFix", "ztt"):
    genTaus=tauReco.findAllGenTaus(mc_particles)
    nGenTaus=len(genTaus)

    loggers["processing"].debug(
        "Found %d gen taus. Details:\n%s",
        nGenTaus,
        "\n".join("GenTau %d: %s" % (i, tau) for i, tau in genTaus.items()),
    )
    gen_taus = True
  else:
    gen_taus = False
    genTaus = {}
    nGenTaus = 0
    loggers["processing"].debug("Skipping gen tau finding for BK sample %s", sample)

  GenZMass=0
  GenZVisMass=0

  if nGenTaus==2:
    ZGenP=genTaus[0].getMomentum()+genTaus[1].getMomentum()
    ZVisPGen=genTaus[0].getvisMomentum()+genTaus[1].getvisMomentum()
    GenZMass=ZGenP.M()
    GenZVisMass=ZVisPGen.M()



  pfos = event.get(pfobjects)
  #hGenVisZMass.Fill(ZVisGen.M(),weight)
  #hGenZMass.Fill(ZGen.M(),weight)
  # TODO adaptar el resto del código para sistemáticos
  recoTau, recoElectrons, recoMuons, recoTau_max, recoTau_min = extractTauDecays(gatr_results_path,
                                                                                  mlpf_results,
                                                                                  eventid,
                                                                                  pfos,
                                                                                  dRMax,
                                                                                  minPTauPhoton,
                                                                                  minPTauPion,
                                                                                  PNeutron,
                                                                                  generalPCut,
                                                                                  photon_config,
                                                                                  test_extremes,
                                                                                  test_pfo,
                                                                                  logger_process)

  recoTaus_extremes = {
      "original": recoTau,
      "min_err": recoTau_min,
      "max_err": recoTau_max
  }
  
  for tree_key in trees:
    # print(f"Processing tree key: {tree_key}")
    # Select reco tau and its variables
    recoTau = recoTaus_extremes[tree_key]
    branches = branches_super[tree_key]
    new_tree = trees[tree_key]
    root_histograms = root_histograms_super[tree_key]
    
    ## get RECO level info
    unsorted_recoTaus = recoTau
    recoTaus= myutils.sort_by_P(unsorted_recoTaus)
    nRecoTaus=len(recoTaus)

    countMuonsP10=0
    countElectronsP10=0
    if nRecoTaus<1:
      logger_process.debug(f"Less than one reco Tau. Skipping event...\n")
      continue

    # Por qué son estos cortes?
    for mu in recoMuons:
      muonP4 = recoMuons[mu].getMomentum()
      if muonP4.P()>10:
        countMuonsP10+=1
        
    for e in recoElectrons:
      electronP4 = recoElectrons[e].getMomentum() 
      if electronP4.P()>10:
        countElectronsP10+=1
    #             ZP4+=electronP4

      # recoMesonP4=recoTaus[0][0]
      # recoTauID=recoTaus[0][1]
      # recoTauQ=recoTaus[0][2]
      # recoTauConsts=recoTaus[0][5]
      # recoPion=recoTauConsts[0]

      # recoPionP4=ROOT.TLorentzVector()
      # recoPionP4.SetXYZM(recoPion.getMomentum().x,recoPion.getMomentum().y,recoPion.getMomentum().z,recoPion.getMass())
    # Select tau
    method = "MaxP"
    if method == "MaxP":
      idx = 0
    else:
      for i, tau in enumerate(recoTaus):
        recoTauID = tau.getID()
        if recoTauID == selectDecay:
          idx = i
          break
    

    recoTau = recoTaus[idx]
    recoTauID = recoTau.getID()
    loggers["processing"].debug("Selected tau with ID %d and selectDecay %d" , recoTauID, selectDecay)
    
    # TODO solo va a considerar taus que tengan 2 fotones -> Decaimiento rho tipico
    # Quizás sería interesante usar el método anterior de que si hay 3 fotones se considere rho
    if recoTauID!=selectDecay:
      loggers["processing"].debug(f"RecoTauID is not selectDecay {selectDecay}. Skipping event...\n")
      continue


    recoMesonP4 = recoTau.getMomentum()
    recoTauID = recoTau.getID()
    recoTauQ = recoTau.getCharge()
    recoTauConsts = recoTau.getDaughters()
    for const in recoTauConsts:
      const_PDG=abs(recoTauConsts[const].getPDG())
      if const_PDG == 211:  # looking for the pion
        recoPion = recoTauConsts[const]
        recoPionP4 = ROOT.TLorentzVector()
        try:
            recoPionP4.SetXYZM(recoPion.getMomentum().x,
                              recoPion.getMomentum().y,
                              recoPion.getMomentum().z,
                              recoPion.getMass())
        except Exception as e:
            recoPionP4.SetXYZM(recoPion.getMomentum().X(),
                              recoPion.getMomentum().Y(),
                              recoPion.getMomentum().Z(),
                              recoPion.getMass())
        loggers["processing"].debug("Found pion with PDG %d and momentum %s", const_PDG, recoPionP4)
        break
    
    if recoMesonP4.P() < tauPCut:
      loggers["processing"].debug(f"TauP < {tauPCut}. Skipping event...\n")
      continue
    # # Corte angular para quitar theta extremo
    # if abs(math.cos(recoMesonP4.Theta()))>0.95:
    #   loggers["processing"].debug(f"RecoMesonTheta > 0.95. Skipping event...\n")
      
    #   continue
    
    # Corte para seleccionar solo desintegraciones lepton-hadron
    if not (
        nRecoTaus == 1 and
        ((countElectronsP10 == 1) ^ (countMuonsP10 == 1))
    ):
      loggers["processing"].debug("Not exactly one reco tau and one lepton with P > 10 GeV. Skipping event...\n")
      continue # Ignoring this event to get the bk
    
    # if (countMuonsP10<1 and countElectronsP10<1):
      
    #   loggers["processing"].debug("No muons or electrons with P > 10 GeV found, skipping event")
    #   continue
    
    is_electron = True if countElectronsP10==1 else False
    lepP4 = None
    lepPDG = 0
    if is_electron:
    # solo hay uno por construcción
      for e in recoElectrons:
          electronP4 = recoElectrons[e].getMomentum()
          if electronP4.P() > 10:
              lepP4 = electronP4
              lepPDG = 11   # electrón
              break
    else:
        for mu in recoMuons:
            muonP4 = recoMuons[mu].getMomentum()
            if muonP4.P() > 10:
                lepP4 = muonP4
                lepPDG = 13   # muón
                break
    
    # Por qué hace este corte?
    # if recoMesonP4.Theta()>1.565 and recoMesonP4.Theta()<1.575 :  # stupid cut to remove high momentum muons at pi/2
      # continue 

    ZMass=0 #ZP4.M()




    genIndex=-1
    # closestDR=10 # find always one
    closestDR=10 # find always one
    for g in range(0,nGenTaus):
        #if genTaus[g][1]!=1: continue
        genP4=genTaus[g].getMomentum()
        dR=myutils.dRAngle(genP4,recoMesonP4)
        if dR<closestDR:
          closestDR=dR
          genIndex=g 

    if genIndex==-1 and gen_taus:
      loggers["processing"].debug("No matching gen tau found. Skipping event...\n")
      continue 
    # print(gen_taus)
    if gen_taus:
      genMesonP4=genTaus[genIndex].getvisMomentum()
      genRhoP4=genMesonP4 # repeat, just make it work, then clean
      genTauID=genTaus[genIndex].getID()
      genTauP4=genTaus[genIndex].getMomentum()
      genTauConst=genTaus[genIndex].getDaughters()
      loggers["processing"].debug("Found gen tau with ID %d and momentum %s", genTauID, genTauP4)
      
      for const in genTauConst:
        const_PDG=abs(genTauConst[const].getPDG())
        if const_PDG == 211:
          genPion = genTauConst[const]
          genPionP4 = ROOT.TLorentzVector()
          try:
            genPionP4.SetXYZM(genPion.getMomentum().x,
                              genPion.getMomentum().y,
                              genPion.getMomentum().z,
                              genPion.getMass())
          except Exception as e:
            genPionP4.SetXYZM(genPion.getMomentum().X(),
                              genPion.getMomentum().Y(),
                              genPion.getMomentum().Z(),
                              genPion.getMass())
          loggers["processing"].debug("Found gen pion with PDG %d and momentum %s", const_PDG, genPionP4)
          break
    else:
      genMesonP4=ROOT.TLorentzVector()
      genMesonP4.SetXYZM(0,0,0,0)
      genTauP4=ROOT.TLorentzVector()
      genTauP4.SetXYZM(0,0,0,0)
      genPionP4=ROOT.TLorentzVector()
      genPionP4.SetXYZM(0,0,0,0)
      genTauID=-1
      genTauConst={}
      genRhoP4=ROOT.TLorentzVector()
      genRhoP4.SetXYZM(0,0,0,0)
    # genPion=genTauConst[0]        
    # genPionP4=ROOT.TLorentzVector()
    # genPionP4.SetXYZM(genPion.getMomentum().x,genPion.getMomentum().y,genPion.getMomentum().z,genPion.getMass())
    if gen_taus:
      (gen_cos_theta,gen_cos_psi,gen_cos_beta,gen_w,weight_P1,weight_M1)=optimalVariabRho.wVariab(genTauP4,
                                                                                                  genMesonP4,genPionP4,beamE, sin_eff=args.sin_eff)
      gen_cos_theta_tau=math.cos(genTauP4.Theta())
    else:
      gen_cos_theta=0
      gen_cos_psi=0
      gen_cos_beta=0
      gen_w=0
      weight_P1=0
      weight_M1=0
      gen_cos_theta_tau=0
    
    (cos_theta,cos_psi,cos_beta,w)=optimalVariabRho.wVariabRECO(recoMesonP4,
                                                                recoPionP4,
                                                                beamE,
                                                                )
    cos_theta_rho=math.cos(recoMesonP4.Theta())

    #(gen_cos_theta,gen_cos_psi,gen_cos_beta,gen_w,weight_P1,weight_M1)=optimalVariabRho.wVariab(genTauP4,genRhoP4,genPionP4,beamE)
    #(cos_theta,cos_psi,cos_beta,w)=optimalVariabRho.wVariabRECO(recoMesonP4,recoPionP4,beamE)
    #gen_cos_theta_tau=math.cos(genTauP4.Theta())
    # Photons information
    if gen_taus:
      nPhotonsGen=0.
      for const in genTauConst:
        pdg = abs(genTauConst[const].getPDG())
        if pdg == 111:
          pi0const = genTauConst[const].getDaughters()
          for photon_const in pi0const:
            if photon_const.getGeneratorStatus()!=1:
              continue
            nPhotonsGen += 1
            photonP4 = ROOT.TLorentzVector()
            try:
              photonP4.SetXYZM(photon_const.getMomentum().x,
                                photon_const.getMomentum().y,
                                photon_const.getMomentum().z,
                                photon_const.getMass())
            except Exception as e:
              photonP4.SetXYZM(photon_const.getMomentum().X(),
                                photon_const.getMomentum().Y(),
                                photon_const.getMomentum().Z(),
                                photon_const.getMass())
            branches["gen_photons_E"].push_back(photonP4.E())
            branches["gen_photons_theta"].push_back(photonP4.Theta())
            branches["gen_photons_phi"].push_back(photonP4.Phi())
    else:
      nPhotonsGen=0.
      branches["gen_photons_E"].push_back(0.)
      branches["gen_photons_theta"].push_back(0.)
      branches["gen_photons_phi"].push_back(0.)
    
    nPhotonsReco=0.
    for const in recoTauConsts:
      pdg = abs(recoTauConsts[const].getPDG())
      if pdg == 22:
        nPhotonsReco += 1
        photon = recoTauConsts[const]
        photonP4 = ROOT.TLorentzVector()
        try:
          photonP4.SetXYZM(photon.getMomentum().x,
                            photon.getMomentum().y,
                            photon.getMomentum().z,
                            photon.getMass())
        except Exception as e:
          photonP4.SetXYZM(photon.getMomentum().X(),
                            photon.getMomentum().Y(),
                            photon.getMomentum().Z(),
                            photon.getMass())
        branches["reco_photons_E"].push_back(photonP4.E())
        branches["reco_photons_theta"].push_back(photonP4.Theta())
        branches["reco_photons_phi"].push_back(photonP4.Phi())

    if abs(cos_theta)==1: continue  # border cases in which the calculation of cosTheta failed
    if abs(cos_psi)==1: continue

    if selectedEvents==prevSelectedEvents:
      selectedEvents+=1
    

    loggers["processing"].debug("Selected event %d with GEN index %d, GEN tau ID %d, and RECO tau ID %d",
        totalEvents, genIndex, genTauID, recoTauID
    )
    branches["genTauP"].value=genTauP4.P()
    branches["genTauTheta"].value=genTauP4.Theta()
    branches["genTauPhi"].value=genTauP4.Phi()
    branches["genTauE"].value=genTauP4.E()
    branches["genTauM"].value=genTauP4.M()
    branches["genMesonP"].value=genMesonP4.P()
    branches["genMesonTheta"].value=genMesonP4.Theta()
    branches["genMesonPhi"].value=genMesonP4.Phi()
    branches["genMesonE"].value=genMesonP4.E()
    branches["genMesonM"].value=genMesonP4.M()
    branches["genPionP"].value=genPionP4.P()
    branches["genPionTheta"].value=genPionP4.Theta()
    branches["genPionPhi"].value=genPionP4.Phi()
    branches["genPionE"].value=genPionP4.E()
    branches["genPionM"].value=genPionP4.M()
    branches["gen_cos_theta"].value=gen_cos_theta
    branches["gen_cos_psi"].value=gen_cos_psi
    branches["gen_cos_beta"].value=gen_cos_beta
    branches["genOmega"].value=gen_w 
    branches["weight_P1"].value=weight_P1
    branches["weight_M1"].value=weight_M1
    branches["genTauID"].value=genTauID
    branches["gen_cos_theta_tau"].value=gen_cos_theta_tau
    branches["recoMesonP"].value=recoMesonP4.P()
    branches["recoMesonTheta"].value=recoMesonP4.Theta()
    branches["recoMesonPhi"].value=recoMesonP4.Phi()
    branches["recoPionP"].value=recoPionP4.P()
    branches["recoPionTheta"].value=recoPionP4.Theta()
    branches["recoPionPhi"].value=recoPionP4.Phi()
    branches["recoMesonE"].value=recoMesonP4.E()
    branches["recoMesonM"].value=recoMesonP4.M()
    branches["recoPionE"].value=recoPionP4.E()
    branches["recoPionM"].value=recoPionP4.M()
    branches["cos_theta"].value=cos_theta
    branches["cos_psi"].value=cos_psi
    branches["cos_beta"].value=cos_beta
    branches["omega"].value=w
    branches["recoTauID"].value=recoTauID
    branches["cos_theta_rho"].value=cos_theta_rho
    branches["ZMass"].value=ZMass
    branches["GenZMass"].value=GenZMass
    branches["GenZVisMass"].value=GenZVisMass
    branches["beamE"].value=beamE
    branches["nPhotonsReco"].value=nPhotonsReco
    branches["nPhotonsGen"].value=nPhotonsGen
    branches["isElectron"].value=float(is_electron)
    branches["lepP"].value=lepP4.P()
    branches["lepE"].value=lepP4.E()
    branches["lepTheta"].value=lepP4.Theta()
    branches["lepPhi"].value=lepP4.Phi()
    branches["lepPDG"].value=lepPDG
    new_tree.Fill()

    root_histograms["Reco"]["Events"]["RecoMesonEOverBeamE_ALL"].Fill(recoMesonP4.E()/beamE,weight)
    
    root_histograms["Reco"]["Events"]["RecoMesonCosTheta_ALL"].Fill(math.cos(recoMesonP4.Theta()),weight)
    root_histograms["Reco"]["Events"]["CosTheta_ALL"].Fill(cos_theta,weight)
    root_histograms["Reco"]["Events"]["CosPsi_ALL"].Fill(cos_psi,weight)
    
    root_histograms["Gen"]["Events"]["Omega_GEN_ALL"].Fill(gen_w,weight)
    root_histograms["Gen"]["Events"]["CosTheta_GEN_ALL"].Fill(gen_cos_theta,weight)
    root_histograms["Gen"]["Events"]["CosPsi_GEN_ALL"].Fill(gen_cos_psi,weight)
    
    root_histograms["Gen"]["Events"]["CosThetaTau_GEN_ALL"].Fill(math.cos(genTauP4.Theta()),weight)
    root_histograms["Gen"]["Events"]["CosThetaRho_GEN_ALL"].Fill(math.cos(genRhoP4.Theta()),weight)
    root_histograms["Gen"]["Events"]["CosThetaRho_ALL"].Fill(math.cos(recoMesonP4.Theta()),weight)


    root_histograms["Reco"]["Events"]["OmegaCosTheta_ALL"].Fill(w,cos_theta_rho,weight)
    root_histograms["Gen"]["Events"]["OmegaCosThetaTau_GEN_ALL"].Fill(gen_w,gen_cos_theta_tau,weight)


    x=2*recoMesonP4.E()/beamE-1
    root_histograms["Reco"]["Events"]["RecoMeson_X_ALL"].Fill(x,weight)

    if genIndex!=-1:
      loggers["processing"].debug("Analysis of GEN tau with index %d, ID %d, and momentum %s", genIndex, genTauID, genTauP4)

      selectGEN=selectDecay
      if selectDecay==2:
          selectGEN=1

      #print("GEN %d %4.2f  %4.2f %4.2f %4.2f %4.2f" %(genTauID,gen_cos_theta,gen_cos_theta_tau,gen_cos_beta, gen_cos_psi,gen_w))
      #print("RECO %d %4.2f %4.2f %4.2f %4.2f %4.2f" %(recoTauID,cos_theta,cos_theta_rho, cos_beta, cos_psi,w))

      if genTauID==selectGEN:
        weight_P1 = weightsPol.newAtauRHO(genTauP4, genMesonP4 , beamE, genTauConst, genTauID, +1)
        weight_M1 = weightsPol.newAtauRHO(genTauP4, genMesonP4 , beamE, genTauConst, genTauID, -1)
        root_histograms["Matched"]["Events"]["MesonEOverBeamE"].Fill(genMesonP4.E()/beamE,weight)
        root_histograms["Matched"]["Events"]["MesonEOverBeamE_P1"].Fill(genMesonP4.E()/beamE,weight_P1*weight)
        root_histograms["Matched"]["Events"]["MesonEOverBeamE_M1"].Fill(genMesonP4.E()/beamE,weight_M1*weight)
        
        root_histograms["Reco"]["Events"]["Omega_SIGNAL"].Fill(w,weight)
        root_histograms["Reco"]["Events"]["Omega_SIGNAL_P1"].Fill(w,weight*weight_P1)
        root_histograms["Reco"]["Events"]["Omega_SIGNAL_M1"].Fill(w,weight*weight_M1)
        
        root_histograms["Gen"]["Events"]["Omega_GEN_SIGNAL"].Fill(gen_w,weight)
        root_histograms["Gen"]["Events"]["Omega_GEN_SIGNAL_P1"].Fill(gen_w,weight*weight_P1)
        root_histograms["Gen"]["Events"]["Omega_GEN_SIGNAL_M1"].Fill(gen_w,weight*weight_M1)
        
        root_histograms["Reco"]["Events"]["OmegaCosTheta_SIGNAL"].Fill(w,cos_theta_rho,weight)
        root_histograms["Reco"]["Events"]["OmegaCosTheta_SIGNAL_P1"].Fill(w,cos_theta_rho,weight*weight_P1)
        root_histograms["Reco"]["Events"]["OmegaCosTheta_SIGNAL_M1"].Fill(w,cos_theta_rho,weight*weight_M1)

        root_histograms["Gen"]["Events"]["OmegaCosThetaTau_GEN_SIGNAL"].Fill(gen_w,gen_cos_theta_tau,weight)
        root_histograms["Gen"]["Events"]["OmegaCosThetaTau_GEN_SIGNAL_P1"].Fill(gen_w,gen_cos_theta_tau,weight*weight_P1)
        root_histograms["Gen"]["Events"]["OmegaCosThetaTau_GEN_SIGNAL_M1"].Fill(gen_w,gen_cos_theta_tau,weight*weight_M1)
    
        root_histograms["Reco"]["Events"]["CosTheta_SIGNAL"].Fill(cos_theta,weight)
        root_histograms["Reco"]["Events"]["CosTheta_SIGNAL_P1"].Fill(cos_theta,weight_M1*weight)
        root_histograms["Reco"]["Events"]["CosTheta_SIGNAL_M1"].Fill(cos_theta,weight_P1*weight)
        
        root_histograms["Gen"]["Events"]["CosTheta_GEN_SIGNAL"].Fill(cos_theta,weight)
        root_histograms["Gen"]["Events"]["CosTheta_GEN_SIGNAL_P1"].Fill(cos_theta,weight_M1*weight)
        root_histograms["Gen"]["Events"]["CosTheta_GEN_SIGNAL_M1"].Fill(cos_theta,weight_P1*weight)
        
          
        root_histograms["Reco"]["Events"]["CosPsi_SIGNAL"].Fill(cos_psi,weight)
        root_histograms["Reco"]["Events"]["CosPsi_SIGNAL_P1"].Fill(cos_psi,weight_M1*weight)
        root_histograms["Reco"]["Events"]["CosPsi_SIGNAL_M1"].Fill(cos_psi,weight_P1*weight)
        
        root_histograms["Gen"]["Events"]["CosPsi_GEN_SIGNAL"].Fill(gen_cos_psi,weight)
        root_histograms["Gen"]["Events"]["CosPsi_GEN_SIGNAL_P1"].Fill(gen_cos_psi,weight_M1*weight)
        root_histograms["Gen"]["Events"]["CosPsi_GEN_SIGNAL_M1"].Fill(gen_cos_psi,weight_P1*weight)
        
        root_histograms["Matched"]["Events"]["MesonCosTheta"].Fill(math.cos(genMesonP4.Theta()), weight)
        root_histograms["Matched"]["Events"]["MesonCosTheta_P1"].Fill(math.cos(genMesonP4.Theta()), weight_P1 * weight)
        root_histograms["Matched"]["Events"]["MesonCosTheta_M1"].Fill(math.cos(genMesonP4.Theta()), weight_M1 * weight)

        root_histograms["Reco"]["Events"]["RecoMesonEOverBeamE_SIGNAL"].Fill(recoMesonP4.E()/beamE, weight)
        root_histograms["Reco"]["Events"]["RecoMesonEOverBeamE_SIGNAL_P1"].Fill(recoMesonP4.E()/beamE, weight_P1 * weight)
        root_histograms["Reco"]["Events"]["RecoMesonEOverBeamE_SIGNAL_M1"].Fill(recoMesonP4.E()/beamE, weight_M1 * weight)

        root_histograms["Reco"]["Events"]["RecoMeson_X"].Fill(x, weight)
        root_histograms["Reco"]["Events"]["RecoMeson_X_P1"].Fill(x, weight * weight_P1)
        root_histograms["Reco"]["Events"]["RecoMeson_X_M1"].Fill(x, weight * weight_M1)

        root_histograms["Reco"]["Events"]["RecoMesonCosTheta_SIGNAL"].Fill(math.cos(recoMesonP4.Theta()), weight)
        root_histograms["Reco"]["Events"]["RecoMesonCosTheta_SIGNAL_P1"].Fill(math.cos(recoMesonP4.Theta()), weight_P1 * weight)
        root_histograms["Reco"]["Events"]["RecoMesonCosTheta_SIGNAL_M1"].Fill(math.cos(recoMesonP4.Theta()), weight_M1 * weight)

        root_histograms["Gen"]["Events"]["MesonType"].Fill(genTauID, weight)
        root_histograms["Reco"]["Events"]["RecoMesonType"].Fill(recoTauID, weight)
        if selectDecay == 2:
          # 2 fotones reco, 2 fotones gen. Reconstruimos masa pi0 reco y hacemos smearing en gen
          cum_P4 = ROOT.TLorentzVector()
          cum_P4.SetXYZM(0, 0, 0, 0)
          for const in recoTauConsts:
            pdg = abs(recoTauConsts[const].getPDG())
            if pdg == 22:
              photon = recoTauConsts[const]
              photonP4 = ROOT.TLorentzVector()
              try:
                photonP4.SetXYZM(photon.getMomentum().x,
                                  photon.getMomentum().y,
                                  photon.getMomentum().z,
                                  photon.getMass())
              except Exception as e:
                photonP4.SetXYZM(photon.getMomentum().X(),
                                  photon.getMomentum().Y(),
                                  photon.getMomentum().Z(),
                                  photon.getMass())
              cum_P4 += photonP4
              
          recoPi0Mass = cum_P4.M()
          root_histograms["Reco"]["Events"]["Pi0Mass_SIGNAL"].Fill(recoPi0Mass)
          cum_P4_gen = ROOT.TLorentzVector()
          cum_P4_gen.SetXYZM(0, 0, 0, 0)
          cum_P4_no_smear = ROOT.TLorentzVector()
          cum_P4_no_smear.SetXYZM(0, 0, 0, 0)
          
          pi0 = None
          for const in genTauConst:
            pdg = abs(genTauConst[const].getPDG())
            if pdg == 111:
              pi0 = genTauConst[const]
              break
          if pi0 is not None:
            for photon in pi0.getDaughters():
                photonP4 = ROOT.TLorentzVector()
                try:
                  photonP4.SetXYZM(photon.getMomentum().x,
                                    photon.getMomentum().y,
                                    photon.getMomentum().z,
                                    photon.getMass())
                except Exception as e:
                  photonP4.SetXYZM(photon.getMomentum().X(),
                                    photon.getMomentum().Y(),
                                    photon.getMomentum().Z(),
                                    photon.getMass())
                photon_E = photonP4.E()
                theta = photonP4.Theta()
                phi   = photonP4.Phi()
                sigma = photon_E*0.16/math.sqrt(photon_E)  # 16% / sqrt(E)
                newE = tauReco.normal_sample(mean=photon_E, stddev=sigma)
                # print("Original E: %f, Smeared E: %f" % (photon_E, newE))
                px = newE * np.sin(theta) * np.cos(phi)
                py = newE * np.sin(theta) * np.sin(phi)
                pz = newE * np.cos(theta)
                smeared_photon_E = ROOT.TLorentzVector()
                smeared_photon_E.SetPxPyPzE(px, py, pz, newE)
          
                cum_P4_gen += smeared_photon_E
                cum_P4_no_smear += photonP4
            genPi0Mass = cum_P4_gen.M()
            # print(f"Gen Pi0 mass: {genPi0Mass}")
            # print(f"Gen Pi0 mass without smearing: {cum_P4_no_smear.M()}")
            # print("\n")
            
            root_histograms["Gen"]["Events"]["Pi0Mass_GEN_SIGNAL"].Fill(genPi0Mass)
      
      else:
        root_histograms["Reco"]["Events"]["RecoMesonEOverBeamE_BG"].Fill(recoMesonP4.E()/beamE, weight)
        root_histograms["Reco"]["Events"]["RecoMesonCosTheta_BG"].Fill(math.cos(recoMesonP4.Theta()), weight)
        root_histograms["Matched"]["Events"]["MesonEOverBeamE_BG"].Fill(genMesonP4.E()/beamE, weight)
        root_histograms["Matched"]["Events"]["MesonCosTheta_BG"].Fill(math.cos(genMesonP4.Theta()), weight)
        root_histograms["Reco"]["Events"]["RecoMeson_X_BG"].Fill(x, weight)

        root_histograms["Gen"]["Events"]["MesonType_BG"].Fill(genTauID, weight)
        root_histograms["Reco"]["Events"]["RecoMesonType_BG"].Fill(recoTauID, weight)

        root_histograms["Reco"]["Events"]["Omega_BG"].Fill(w, weight)
        root_histograms["Reco"]["Events"]["CosTheta_BG"].Fill(cos_theta, weight)
        root_histograms["Reco"]["Events"]["CosPsi_BG"].Fill(cos_psi, weight)
        root_histograms["Reco"]["Events"]["OmegaCosTheta_BG"].Fill(w, cos_theta_rho, weight)
        root_histograms["Gen"]["Events"]["OmegaCosThetaTau_GEN_BG"].Fill(gen_w, gen_cos_theta_tau, weight)

        root_histograms["Gen"]["Events"]["Omega_GEN_BG"].Fill(gen_w, weight)
        root_histograms["Gen"]["Events"]["CosTheta_GEN_BG"].Fill(gen_cos_theta, weight)
        root_histograms["Gen"]["Events"]["CosPsi_GEN_BG"].Fill(gen_cos_psi, weight)

      

        if genTauID==-13:
          root_histograms["Reco"]["Events"]["RecoMesonEOverBeamE_BGMuon"].Fill(recoMesonP4.E()/beamE, weight)
          root_histograms["Matched"]["Events"]["MesonEOverBeamE_BGMuon"].Fill(genMesonP4.E()/beamE, weight)
          root_histograms["Reco"]["Events"]["RecoMesonE_BGMuon"].Fill(recoMesonP4.E(), weight)

          root_histograms["Reco"]["Events"]["RecoMesonCosTheta_BGMuon"].Fill(math.cos(recoMesonP4.Theta()), weight)
          root_histograms["Matched"]["Events"]["MesonCosTheta_BGMuon"].Fill(math.cos(genMesonP4.Theta()), weight)

          root_histograms["Reco"]["Events"]["RecoMeson_BGMuon_PhiTheta"].Fill(recoMesonP4.Theta(), recoMesonP4.Phi(), weight)
          root_histograms["Reco"]["Events"]["RecoMeson_BGMuon_PtTheta"].Fill(recoMesonP4.Theta(), recoMesonP4.Pt(), weight)

          root_histograms["Reco"]["Events"]["Omega_BGMuon"].Fill(w, weight)
          root_histograms["Reco"]["Events"]["OmegaCosTheta_BGMuon"].Fill(w, cos_theta_rho, weight)
          root_histograms["Reco"]["Events"]["CosTheta_BGMuon"].Fill(cos_theta, weight)
          root_histograms["Reco"]["Events"]["CosPsi_BGMuon"].Fill(cos_psi, weight)

          root_histograms["Gen"]["Events"]["OmegaCosThetaTau_GEN_BGMuon"].Fill(gen_w, gen_cos_theta_tau, weight)
          root_histograms["Gen"]["Events"]["Omega_GEN_BGMuon"].Fill(gen_w, weight)
          root_histograms["Gen"]["Events"]["CosTheta_GEN_BGMuon"].Fill(gen_cos_theta, weight)
          root_histograms["Gen"]["Events"]["CosPsi_GEN_BGMuon"].Fill(gen_cos_psi, weight)


        elif genTauID==-11:
          root_histograms["Reco"]["Events"]["RecoMesonEOverBeamE_BGEle"].Fill(recoMesonP4.E()/beamE, weight)
          root_histograms["Matched"]["Events"]["MesonEOverBeamE_BGEle"].Fill(genMesonP4.E()/beamE, weight)
          root_histograms["Reco"]["Events"]["RecoMesonCosTheta_BGEle"].Fill(math.cos(recoMesonP4.Theta()), weight)
          root_histograms["Matched"]["Events"]["MesonCosTheta_BGEle"].Fill(math.cos(genMesonP4.Theta()), weight)

          root_histograms["Reco"]["Events"]["RecoMeson_BGEle_PhiTheta"].Fill(recoMesonP4.Theta(), recoMesonP4.Phi(), weight)
          root_histograms["Reco"]["Events"]["RecoMeson_BGEle_PtTheta"].Fill(recoMesonP4.Theta(), recoMesonP4.Pt(), weight)

          root_histograms["Reco"]["Events"]["Omega_BGEle"].Fill(w, weight)
          root_histograms["Reco"]["Events"]["OmegaCosTheta_BGEle"].Fill(w, cos_theta_rho, weight)
          root_histograms["Reco"]["Events"]["CosTheta_BGEle"].Fill(cos_theta, weight)
          root_histograms["Reco"]["Events"]["CosPsi_BGEle"].Fill(cos_psi, weight)

          root_histograms["Gen"]["Events"]["OmegaCosThetaTau_GEN_BGEle"].Fill(gen_w, gen_cos_theta_tau, weight)
          root_histograms["Gen"]["Events"]["Omega_GEN_BGEle"].Fill(gen_w, weight)
          root_histograms["Gen"]["Events"]["CosTheta_GEN_BGEle"].Fill(gen_cos_theta, weight)
          root_histograms["Gen"]["Events"]["CosPsi_GEN_BGEle"].Fill(gen_cos_psi, weight)


        elif genTauID==0:
          root_histograms["Reco"]["Events"]["RecoMesonEOverBeamE_BGPion"].Fill(recoMesonP4.E()/beamE, weight)
          root_histograms["Matched"]["Events"]["MesonEOverBeamE_BGPion"].Fill(genMesonP4.E()/beamE, weight)

          root_histograms["Reco"]["Events"]["RecoMesonCosTheta_BGPion"].Fill(math.cos(recoMesonP4.Theta()), weight)
          root_histograms["Matched"]["Events"]["MesonCosTheta_BGPion"].Fill(math.cos(genMesonP4.Theta()), weight)

          root_histograms["Reco"]["Events"]["Omega_BGPion"].Fill(w, weight)
          root_histograms["Reco"]["Events"]["OmegaCosTheta_BGPion"].Fill(w, cos_theta_rho, weight)

          root_histograms["Reco"]["Events"]["CosTheta_BGPion"].Fill(cos_theta, weight)
          root_histograms["Reco"]["Events"]["CosPsi_BGPion"].Fill(cos_psi, weight)

          root_histograms["Gen"]["Events"]["OmegaCosThetaTau_GEN_BGPion"].Fill(gen_w, gen_cos_theta_tau, weight)
          root_histograms["Gen"]["Events"]["Omega_GEN_BGPion"].Fill(gen_w, weight)
          root_histograms["Gen"]["Events"]["CosTheta_GEN_BGPion"].Fill(gen_cos_theta, weight)
          root_histograms["Gen"]["Events"]["CosPsi_GEN_BGPion"].Fill(gen_cos_psi, weight)


        elif genTauID==1:
          root_histograms["Reco"]["Events"]["RecoMesonEOverBeamE_BGRho"].Fill(recoMesonP4.E()/beamE, weight)
          root_histograms["Matched"]["Events"]["MesonEOverBeamE_BGRho"].Fill(genMesonP4.E()/beamE, weight)

          root_histograms["Reco"]["Events"]["RecoMesonCosTheta_BGRho"].Fill(math.cos(recoMesonP4.Theta()), weight)
          root_histograms["Matched"]["Events"]["MesonCosTheta_BGRho"].Fill(math.cos(genMesonP4.Theta()), weight)

          root_histograms["Reco"]["Events"]["Omega_BGRho"].Fill(w, weight)
          root_histograms["Reco"]["Events"]["OmegaCosTheta_BGRho"].Fill(w, cos_theta_rho, weight)

          root_histograms["Reco"]["Events"]["CosTheta_BGRho"].Fill(cos_theta, weight)
          root_histograms["Reco"]["Events"]["CosPsi_BGRho"].Fill(cos_psi, weight)

          root_histograms["Gen"]["Events"]["OmegaCosThetaTau_GEN_BGRho"].Fill(gen_w, gen_cos_theta_tau, weight)
          root_histograms["Gen"]["Events"]["Omega_GEN_BGRho"].Fill(gen_w, weight)
          root_histograms["Gen"]["Events"]["CosTheta_GEN_BGRho"].Fill(gen_cos_theta, weight)
          root_histograms["Gen"]["Events"]["CosPsi_GEN_BGRho"].Fill(gen_cos_psi, weight)


        elif genTauID==10:
          root_histograms["Reco"]["Events"]["RecoMesonEOverBeamE_BGA1"].Fill(recoMesonP4.E()/beamE, weight)
          root_histograms["Matched"]["Events"]["MesonEOverBeamE_BGA1"].Fill(genMesonP4.E()/beamE, weight)

          root_histograms["Reco"]["Events"]["RecoMesonCosTheta_BGA1"].Fill(math.cos(recoMesonP4.Theta()), weight)
          root_histograms["Matched"]["Events"]["MesonCosTheta_BGA1"].Fill(math.cos(genMesonP4.Theta()), weight)

          root_histograms["Reco"]["Events"]["Omega_BGA1"].Fill(w, weight)
          root_histograms["Reco"]["Events"]["OmegaCosTheta_BGA1"].Fill(w, cos_theta_rho, weight)

          root_histograms["Reco"]["Events"]["CosTheta_BGA1"].Fill(cos_theta, weight)
          root_histograms["Reco"]["Events"]["CosPsi_BGA1"].Fill(cos_psi, weight)

          root_histograms["Gen"]["Events"]["OmegaCosThetaTau_GEN_BGA1"].Fill(gen_w, gen_cos_theta_tau, weight)
          root_histograms["Gen"]["Events"]["Omega_GEN_BGA1"].Fill(gen_w, weight)
          root_histograms["Gen"]["Events"]["CosTheta_GEN_BGA1"].Fill(gen_cos_theta, weight)
          root_histograms["Gen"]["Events"]["CosPsi_GEN_BGA1"].Fill(gen_cos_psi, weight)


        else:
          root_histograms["Reco"]["Events"]["RecoMesonEOverBeamE_BGOther"].Fill(recoMesonP4.E()/beamE, weight)
          root_histograms["Matched"]["Events"]["MesonEOverBeamE_BGOther"].Fill(genMesonP4.E()/beamE, weight)

          root_histograms["Reco"]["Events"]["RecoMesonCosTheta_BGOther"].Fill(math.cos(recoMesonP4.Theta()), weight)
          root_histograms["Matched"]["Events"]["MesonCosTheta_BGOther"].Fill(math.cos(genMesonP4.Theta()), weight)

          root_histograms["Reco"]["Events"]["Omega_BGOther"].Fill(w, weight)
          root_histograms["Reco"]["Events"]["OmegaCosTheta_BGOther"].Fill(w, cos_theta_rho, weight)

          root_histograms["Reco"]["Events"]["CosTheta_BGOther"].Fill(cos_theta, weight)
          root_histograms["Reco"]["Events"]["CosPsi_BGOther"].Fill(cos_psi, weight)

          root_histograms["Gen"]["Events"]["OmegaCosThetaTau_GEN_BGOther"].Fill(gen_w, gen_cos_theta_tau, weight)
          root_histograms["Gen"]["Events"]["Omega_GEN_BGOther"].Fill(gen_w, weight)
          root_histograms["Gen"]["Events"]["CosTheta_GEN_BGOther"].Fill(gen_cos_theta, weight)
          root_histograms["Gen"]["Events"]["CosPsi_GEN_BGOther"].Fill(gen_cos_psi, weight)



    else:
      root_histograms["Reco"]["Events"]["RecoMesonEOverBeamE_BG"].Fill(recoMesonP4.E()/beamE, weight)
      root_histograms["Reco"]["Events"]["RecoMesonCosTheta_BG"].Fill(math.cos(recoMesonP4.Theta()), weight)

      root_histograms["Matched"]["Events"]["MesonEOverBeamE_BG"].Fill(0, weight)
      root_histograms["Matched"]["Events"]["MesonType_BG"].Fill(-3, weight)
      root_histograms["Reco"]["Events"]["RecoMesonType_BG"].Fill(recoTauID, weight)

      root_histograms["Reco"]["Events"]["Omega_BG"].Fill(w, weight)
      root_histograms["Reco"]["Events"]["OmegaCosTheta_BG"].Fill(w, cos_theta_rho, weight)
      root_histograms["Gen"]["Events"]["OmegaCosThetaTau_GEN_BG"].Fill(gen_w, gen_cos_theta_tau, weight)

      root_histograms["Reco"]["Events"]["CosTheta_BG"].Fill(cos_theta, weight)
      root_histograms["Reco"]["Events"]["CosPsi_BG"].Fill(cos_psi, weight)

      root_histograms["Gen"]["Events"]["Omega_GEN_BG"].Fill(gen_w, weight)
      root_histograms["Gen"]["Events"]["CosTheta_GEN_BG"].Fill(gen_cos_theta, weight)
      root_histograms["Gen"]["Events"]["CosPsi_GEN_BG"].Fill(gen_cos_psi, weight)


output_config_file = general_configs["outputpath"] + "config.yaml"
with open(output_config_file, "w") as file:
    yaml.dump(general_configs["config"], file)
    loggers["io"].info("Configuration file saved to %s", output_config_file)


loggers["io"].info(f"Run over {totalEvents}, selected {selectedEvents}")#," ->",selectedEvents/totalEvents)
loggers["io"].info(f"Weights? {sumWeights}, {sumWeightsP1}, {sumWeightsM1}")
loggers["io"].info(f"Writing file {fileOutName}")


results_dict = {"TotalEvents": totalEvents,
                "SelectedEvents": selectedEvents,
                "SumWeights": sumWeights,
                "SumWeightsP1": sumWeightsP1,
                "SumWeightsM1": sumWeightsM1}
outfile.cd() # =ROOT.TFile(fileOutName,"RECREATE")
for tree_key in trees:
  write_histograms_recursive(root_histograms_super[tree_key])
  trees[tree_key].Write()
results_df = pd.DataFrame(results_dict, index=[0])
decay_str = general_configs["decay"]
outpath_df = general_configs["outputpath"] + f"results_summary_{decay_str}.csv"
results_df.to_csv(outpath_df, index=False)


outfile.Close()


