import sys
import math
import ROOT
from array import array
from podio import root_io
import edm4hep
from modules.ParticleObjects import GenParticle, RecoParticle

from modules import myutils

import logging
try:
   logger = logging.getLogger("processing")
except:
   logger = None

import ROOT

# ----------------------------------------------------------------------
# Utilidad: recorrer recursivamente las hijas de un tau y recoger productos finales
# ----------------------------------------------------------------------
def GetGenTauDecayProducts(mc_particles, only_final_state=True):
    """
    Recorre todos los taus generadores (status==2) y obtiene sus productos de decaimiento.
    Si only_final_state=True, devuelve solo partículas finales (status==1).
    
    Returns:
        genProds (dict): idx -> GenParticle (de tu clase), conteniendo 4-momento y PDG de cada producto.
    """
    genProds = {}
    n = 0
    for particle in mc_particles:
        # Solo taus generadores finales (para no contar duplicados)
        if abs(particle.getPDG()) != 15:
            continue
        if particle.getGeneratorStatus() != 2:
            continue

        stack = list(particle.getDaughters())
        while stack:
            d = stack.pop()
            d_status = getattr(d, "getGeneratorStatus", lambda: None)()
            d_pdg = abs(d.getPDG())

            # Si queremos solo productos finales (status==1), seguimos bajando si no lo son
            if only_final_state and d_status is not None and d_status != 1:
                # seguir explorando descendencia
                for dd in d.getDaughters():
                    stack.append(dd)
                continue

            dauP4 = ROOT.TLorentzVector()
            dauP4.SetXYZM(d.getMomentum().x, d.getMomentum().y, d.getMomentum().z, d.getMass())
            genPart = GenParticle(dauP4, d_pdg, d.getCharge(), dauP4, 0, 0, None, d_pdg)
            genProds[n] = genPart
            n += 1

    return genProds


# ----------------------------------------------------------------------
# Reco: fotones
# ----------------------------------------------------------------------
def GetRecoPhotons(pfos):
    """
    Obtiene los fotones reconstruidos (PDG=22) de la colección de PFOs.
    
    Returns:
        recoPhotons (dict): idx -> RecoParticle (o el objeto original si falla el wrap)
    """
    recoPhotons = {}
    nRecoPhotons = 0
    for particle in pfos:
        if abs(particle.getPDG()) != 22:
            continue
        try:
            dauP4 = ROOT.TLorentzVector()
            dauP4.SetXYZM(particle.getMomentum().x, particle.getMomentum().y, particle.getMomentum().z, particle.getMass())
            # carga 0 para fotón
            recoPhoton = RecoParticle(dauP4, 22, 0, 0, 0, None, 22)
        except AttributeError:
            # por si ya viene con la interfaz adecuada
            recoPhoton = particle

        recoPhotons[nRecoPhotons] = recoPhoton
        nRecoPhotons += 1

    return recoPhotons


# ----------------------------------------------------------------------
# Matching: fotón reco vs partícula gen (priorizando fotones gen)
# ----------------------------------------------------------------------
def MatchRecoPhotonGenParticle(genProds, recoPhotons, maxDRMatch=1.0, non_considered_particles=[], force = False):
    """
    Empareja cada fotón reconstruido con el producto generador más cercano (dR),
    priorizando que el emparejamiento sea con un fotón (PDG=22). Evita matches 1:N.
    
    Args:
        genProds (dict): productos de decaimiento del tau a nivel gen (usar GetGenTauDecayProducts).
        recoPhotons (dict): fotones reconstruidos (usar GetRecoPhotons).
        maxDRMatch (float): umbral de dR máximo aceptado.
        non_considered_particles (list[int]): PDGs a ignorar (p.ej., neutrinos [12,14,16]).
    
    Returns:
        dict: reco_idx -> gen_idx
    """
    reco_gen_match = {}

    for recoIdx in recoPhotons:
        findMatch = -1           # mejor match gen (cualquier PDG)
        findMatchPhoton = -1     # mejor match gen pero fotón PDG=22
        recoP4 = recoPhotons[recoIdx].getMomentum()

        minDR = maxDRMatch
        minDRPhoton = maxDRMatch

        for genIdx in genProds:
            genP4 = genProds[genIdx].getMomentum()
            angleMatch = myutils.dRAngle(recoP4, genP4)

            genPDG = abs(genProds[genIdx].getPDG())
            if genPDG in non_considered_particles:
                continue

            # match global
            if angleMatch < minDR:
                minDR = angleMatch
                findMatch = genIdx

            # priorizar fotón si existe
            if genPDG == 22 and angleMatch < minDRPhoton:
                minDRPhoton = angleMatch
                findMatchPhoton = genIdx

        # evita que dos reco apunten al mismo gen
        if findMatchPhoton != -1 and findMatchPhoton not in reco_gen_match.values() and force:
            reco_gen_match[recoIdx] = findMatchPhoton
        elif findMatch != -1 and findMatch not in reco_gen_match.values():
            reco_gen_match[recoIdx] = findMatch

    return reco_gen_match


# ----------------------------------------------------------------------
# Wrapper alto nivel: listas de matched / unmatched para fotones
# ----------------------------------------------------------------------
def MatchedUnmatchedPhotons(mc_particles, pfos, maxDRMatch=1.0, non_considered_particles=None):
    """
    Devuelve fotones gen sin emparejar, fotones reco sin emparejar,
    dict de reco->gen cuando el gen es fotón y dict de reco->gen cuando el gen es otra partícula.
    """
    if non_considered_particles is None:
        non_considered_particles = []  # p.ej. [12, 14, 16] si quieres ignorar neutrinos

    # Productos de decaimiento del τ (final state) -> contendrá fotones gen si los hay
    genProds = GetGenTauDecayProducts(mc_particles, only_final_state=True)
    # Fotones reconstruidos
    recoPhotons = GetRecoPhotons(pfos)

    matched = MatchRecoPhotonGenParticle(genProds, recoPhotons, maxDRMatch, non_considered_particles)

    # Clasificación según PDG del generador emparejado
    reco_photons_matched_with_gen_photons = {}
    reco_photons_matched_with_other_particles = {}

    for recoIdx, genIdx in matched.items():
        reco_obj = recoPhotons[recoIdx]
        gen_obj = genProds[genIdx]
        if abs(gen_obj.getPDG()) == 22:
            reco_photons_matched_with_gen_photons[reco_obj] = gen_obj
        else:
            reco_photons_matched_with_other_particles[reco_obj] = gen_obj

    # No emparejados en gen (solo fotones gen)
    unmatched_gen_photons = []
    # Solo contamos gen fotones (22) no usados en el matching
    matched_gen_indices = set(matched.values())
    for genIdx in genProds:
        gen_obj = genProds[genIdx]
        if abs(gen_obj.getPDG()) == 22 and genIdx not in matched_gen_indices:
            unmatched_gen_photons.append(gen_obj)

    # No emparejados en reco (todos los reco phot)
    unmatched_reco_photons = []
    for recoIdx in recoPhotons:
        if recoIdx not in matched.keys():
            unmatched_reco_photons.append(recoPhotons[recoIdx])

    return (unmatched_gen_photons,
            unmatched_reco_photons,
            reco_photons_matched_with_gen_photons,
            reco_photons_matched_with_other_particles)

   
def GetGenDaughters(mc_particles):
   """Find all generator level pion.
   
   Args:
       mc_particles (Particle Collection): All particles in the event.
       
   Returns:
      genPions (dict): Dictionary with the generator level pions containing tuples with the visible 4-momentum, the tau ID, and the charge.
   """
   genDaughters={}
   nGenDau=0
   for particle in mc_particles:
      # only taus
      if abs(particle.getPDG()) != 15:
         continue
      # in the pythia sample we need to check the genStatus:
      # (in some events we have several copies of the tau)
      
      # 2: final state tau (to not double count)
      if particle.getGeneratorStatus()!=2:
         continue
#        print ("genTau!",particle.getGeneratorStatus())
      daughters=particle.getDaughters()
      
      # loop over daughter particles of the tau 
      for dTau in daughters:
            dauPDG=abs(dTau.getPDG())
            dauP4=ROOT.TLorentzVector()
            dauP4.SetXYZM(dTau.getMomentum().x,dTau.getMomentum().y,dTau.getMomentum().z,dTau.getMass())
            genDau = GenParticle(dauP4, dauPDG, dTau.getCharge(), dauP4, 0, 0, None, dauPDG)
            genDaughters[nGenDau]=genDau
            nGenDau+=1

   return genDaughters

def GetRecoPions(pfos):
   """Find all reco level pion.
   
   Args:
       mc_particles (Particle Collection): All particles in the event.
       
   Returns:
      genPions (dict): Dictionary with the generator level pions containing tuples with the visible 4-momentum, the tau ID, and the charge.
   """
   recoPions={}
   nRecoPions=0
   for particle in pfos:
      # only pions
      if abs(particle.getPDG()) != 211:
         continue
      
      try:
         dauP4=ROOT.TLorentzVector()
         dauP4.SetXYZM(particle.getMomentum().x,particle.getMomentum().y,particle.getMomentum().z,particle.getMass())
         recoPion = RecoParticle(dauP4, 211, particle.getCharge(),0, 0, None, 211)
      except AttributeError:
         recoPion = particle

      recoPions[nRecoPions]=recoPion
      nRecoPions+=1

   return recoPions

def MatchRecoPionGenParticle(genDaus, recoPions, maxDRMatch=1, non_considered_particles = [], force = False):
   reco_gen_match = {}
   for recoPion in recoPions:
      findMatch = -1
      findMatchPion = -1
      recoPiP4 = recoPions[recoPion].getMomentum()
      minDR = maxDRMatch
      minDRPion = maxDRMatch
      for genDau in genDaus:
         genP4 = genDaus[genDau].getMomentum()
         angleMatch = myutils.dRAngle(recoPiP4, genP4)
         # HACER QUE NO PUEDA SER UN NEUTRINO
         # HACER QUE SEA UNA PARTÍCULA CARGADA (METER CONDICIONES)
         gendauPDG = abs(genDaus[genDau].getPDG())
         if (gendauPDG in non_considered_particles):
            continue
         # logger.info(f"GENPDG INTENTA MATCH{gendauPDG}")
         if angleMatch < minDR:
            minDR = angleMatch
            findMatch = genDau
         if gendauPDG == 211 and angleMatch < minDRPion:
            minDRPion = angleMatch
            findMatchPion = genDau
            
      if findMatchPion != -1 and findMatchPion not in reco_gen_match.values() and force:
         # Prioriza el match con pión
         reco_gen_match[recoPion] = findMatchPion
      elif findMatch != -1 and findMatch not in reco_gen_match.values():
         # Si no hay match con pión, usa la partícula más cercana
         reco_gen_match[recoPion] = findMatch
         
   return reco_gen_match
      

def MatchedUnmatchedPions(mc_particles, pfos, maxDRMatch=1, non_considered_particles=[]):
   """ Get matched and unmatched pions from generator and reconstructed particles.
   Args:
       mc_particles (Particle Collection): All particles in the event.
       pfos (Particle Collection): All particles in the event.
       maxDRMatch (float, optional): Maximum angle between the momenta. Defaults to 1.
   Returns:
       Tuple: Tuple with dictionaries of matched and unmatched pions.
   """
   genDaus = GetGenDaughters(mc_particles)
   recoPions = GetRecoPions(pfos)
   
   matched_pions = MatchRecoPionGenParticle(genDaus, recoPions, maxDRMatch, non_considered_particles)
   
   reco_pions_matched_with_gen_pions = {}
   reco_pions_matched_with_other_particles = {}
   for recoPion, genDau in matched_pions.items():
      recoPion_obj = recoPions[recoPion]
      genDau_obj = genDaus[genDau]
      genDauPDG = genDau_obj.getPDG()
      if genDauPDG == 211:
         reco_pions_matched_with_gen_pions[recoPion_obj] = genDau_obj
      else:
         reco_pions_matched_with_other_particles[recoPion_obj] = genDau_obj
   # unmatched generator pions
   unmatched_gen_pions = []
   for genDau in genDaus:
      if genDau not in matched_pions.values() and genDaus[genDau].getPDG() == 211:
         unmatched_gen_pions.append(genDaus[genDau])
   # unmatched reconstructed pions
   unmatched_reco_pions = []
   for recoPion in recoPions:
      if recoPion not in matched_pions.keys():
         unmatched_reco_pions.append(recoPions[recoPion])
         
   return unmatched_gen_pions, unmatched_reco_pions, reco_pions_matched_with_gen_pions, reco_pions_matched_with_other_particles
      

def MatchRecoGenTau(genTau, recoTaus, nTausType, maxDRMatch=1, selectDecay=-777):
   """ Find the reconstructed tau that is closest to the generator level tau using the angle between the momenta.
      Args:
            genTau: generator level tau.
            recoTaus: list of reconstructed taus.
            maxDRMatch: maximum angle between the momenta.
      Returns:
            Tuple: Tuple with the index of the closest reconstructed tau and the number of taus of the same type.
      """
   findMatch=-1
   genVisTauP4 = genTau.getvisMomentum()
   nRecoTaus = len(recoTaus)
   
   for j in range(0,nRecoTaus):
      recoTauP4=recoTaus[j].getMomentum()
      recoTauId=recoTaus[j].getID()

      # we want to study migrations: keep all the decays but count how many are good 
      # careful, at reco level we count photons and at gen level pi0s: difference in the
      # decay mode (1 gen can be 1,2 reco)

      recoDM=recoTauId
      if recoTauId==2:
         recoDM=1
      elif (recoTauId>=11 and recoTauId<15):
         recoDM=11
      elif recoTauId>=3 and recoTauId<10:
         recoDM=3

      if selectDecay!=-777 and selectDecay==recoDM:
            nTausType+=1

      # but remove at least the leptonic ones / failed ID
      # if recoTauId<0:
      #    continue

      angleMatch=myutils.dRAngle(recoTauP4, genVisTauP4)

      # find closest
      if angleMatch<maxDRMatch:
         maxDRMatch=angleMatch
         findMatch=j
      # if logger:
      #    logger.debug(
      #       f"genTau: {genVisTauP4.P()}, recoTau: {recoTauP4.P()} id {recoDM} idx {j}, angleMatch: {angleMatch}, maxDRMatch: {maxDRMatch}, match: {findMatch}"
      #    )
         
   return findMatch, nTausType

# Check a generator level tau candidate, find the decay, 
# and compute visible (meson) variables 
def visTauGen(candTau):
   """ Check a generator level tau candidate, find the decay, and compute visible (meson) variables.

   Args:
       candTau (Particle ObjefindAllGenZs
   Returns:
       Tuple: Tuple with the visible 4-momentum, the tau ID, the charge, the true 4-momentum, the maximum angle between constituents, the number of constituents, and the constituents.
   """
   countPionsTauGen=0
   countPi0TauGen=0
   countMuonDecay=0
   countElectronDecay=0
   countOther=0

   genTauP4=ROOT.TLorentzVector()
   
   genTauP4.SetXYZM(candTau.getMomentum().x,candTau.getMomentum().y,candTau.getMomentum().z,candTau.getMass())

   # visible 4-momentum
   visTauP4=ROOT.TLorentzVector()
   visTauP4.SetXYZM(0,0,0,0)
   chargeTau=0
   daughters=candTau.getDaughters()
   tauID=-1

   maxAngleConsts=0
   nConsts=0
   const={}

   # loop over daughter particles of the tau 
   for dTau in daughters:
         dauP4=ROOT.TLorentzVector()
         dauP4.SetXYZM(dTau.getMomentum().x,dTau.getMomentum().y,dTau.getMomentum().z,dTau.getMass())
         dauPDG=abs(dTau.getPDG())

         #print ('...dau',dauP4.P(),dauP4.Theta(),dauP4.Phi(),dau.getMass())

         # we want to compare the reco P4 to the 'visible' gen P4: skip neutrinos
         # PDG ID of Neutrinos

         if (dauPDG==12 or dauPDG==14 or dauPDG==16):
            continue 

         # lepton decays 
         if dauPDG==13 :
            countMuonDecay+=1
            #continue # either filter here or at the analysis level
         if dauPDG==11 :
            countElectronDecay+=1               
            #continue # either filter here or at the analysis level

         # in this Pythia sample the tau decay directly goes to pi0/pi, without the rho/a1
         # to be checked in KKMC and Whizard...
         # if there was a rho, we would need an additional step
         
         # 211 -> Pions 321 -> Kaons 323 -> K*(892) 111 -> Pi0
         if dauPDG==211 or dauPDG==321 or dauPDG==323:   # kaons and pions paired together 
            countPionsTauGen+=1
         elif dauPDG==111 :
            countPi0TauGen+=1
         # Charged particles that are not electrons or muons
         elif dTau.getCharge()!=0 and (dauPDG!=11 and dauPDG!=13):
            logger.warning(
               f"Found a charged particle with PDG {dauPDG} and charge {dTau.getCharge()} in the tau decay. "
               f"This is not expected and may indicate an issue with the tau decay reconstruction."
            )
            countOther+=1

         # compute the angle of the constituents (cone size) for further studies 
         dR=myutils.dRAngle(genTauP4,dauP4)
         if maxAngleConsts<dR:
               maxAngleConsts=dR

         const[nConsts]=dTau
         nConsts+=1

         # Sum the visible 4-momentum and charge
         chargeTau+=dTau.getCharge()
         visTauP4+=dauP4

   # now encode the ID in a int
   # this could be much more elegant, simple for now
   if countMuonDecay>0:
      tauID=-13
   elif countElectronDecay>0:
      tauID=-11
   elif countOther>0: # refinement: check what these are 
      tauID=-2
   elif abs(chargeTau)==1: 
      if (countPionsTauGen==1):
               tauID=countPi0TauGen
      elif (countPionsTauGen==3):
               tauID=countPi0TauGen+10

   # return an object with the visible pt, ID, charge, and the true Pt 
   # a future step would be to define a class for the tau
   if tauID == 0:
      logger.debug(f"Tau Visible Momentum: {visTauP4.P()}")
      cum_momentum = ROOT.TLorentzVector()
      cum_momentum.SetXYZM(0, 0, 0, 0)
      for const_key in const:
         daup4 = ROOT.TLorentzVector()
         daup4.SetXYZM(const[const_key].getMomentum().x, const[const_key].getMomentum().y, const[const_key].getMomentum().z, const[const_key].getMass())
         cum_momentum += daup4
         logger.debug(f"Constituent {const_key} PDG {const[const_key].getPDG()}: {daup4.P()}")
         logger.debug(f"Total Visible Momentum: {cum_momentum.P()}")
   return (visTauP4,tauID,chargeTau,genTauP4,maxAngleConsts,nConsts,const)                 

# Reversed procedure for reconstructed pfos
# Starting from a pion, find particles in a cone around it, and 
# build the tau 
def buildTauFromPion(lead, allPfs, DRCone=1, minP_photon=0, minP_pion=0, PNeutron=1, genminP = 0.5, charge_condition=True):
   """ Starting from a pion, find particles in a cone around it, and build the tau.

   Args:
      lead (Particle Object): Pion candidate.
      allPfs (Particle Collection): All particles in the event.
      DRCone (int, optional): Radius of the cone. Defaults to 1.
      minP_photon (int, optional): Minimum photon momentum. Defaults to 0.
      minP_pion (int, optional): Minimum pion momentum. Defaults to 0.
      PNeutron (int, optional): Minimum neutron momentum. Defaults to 10.
      genminP (int, optional): Minimum general level momentum. Defaults to 0.5.
   
   Returns:
      Tuple: Tuple with the 4-momentum of the tau, the tau ID, the charge, the maximum angle between constituents, the number of constituents, and the constituents.
   """
   countPions=1
   countPhotons=0

   # Initialize charge with the pion charge
   chargeTau=lead.getCharge()
   tauID=-1

   # Initialize the 4-momentum of the tau with the pion 4-momentum
   leadP4=ROOT.TLorentzVector()
   try:
      leadP4.SetXYZM(lead.getMomentum().x,lead.getMomentum().y,lead.getMomentum().z,lead.getMass())
   except AttributeError:
      leadP4.SetXYZM(lead.getMomentum().X(),lead.getMomentum().Y(),lead.getMomentum().Z(),lead.getMass())
   tauP4=ROOT.TLorentzVector()
   tauP4=leadP4

   maxConeTau=0
   # Constituents of the tau
   const={}

   nConsts=1
   const[0]=lead      
   countNeutrons=0

#        print ('...lead',leadP4.P(),leadP4.Theta(),math.cos(leadP4.Theta())) # leadP4.Phi(),lead.getMass())
   # Set to avoid duplicates
   # found_pions_id = set()
   for cand in allPfs:
      if (cand==lead):
         continue 

      candP4=ROOT.TLorentzVector()
      try:
         candP4.SetXYZM(cand.getMomentum().x,cand.getMomentum().y,cand.getMomentum().z,cand.getMass())
      except AttributeError:
         candP4.SetXYZM(cand.getMomentum().X(),cand.getMomentum().Y(),cand.getMomentum().Z(),cand.getMass())

      candPDG=abs(cand.getPDG())

#             print ('...cand',candP4.P(),candP4.Theta(),candP4.Phi(),cand.getMass())

      # max angle? check backgrounds as well
      dR=myutils.dRAngle(candP4,leadP4) 
      if (dR>DRCone):  # cut to be tuned
         continue 
      # how low should we go in P?
      if (candP4.P()<genminP):
         continue 

      # now check ID and clean
      # Ignore events with electrons and muons
      if abs(candPDG)==11 or abs(candPDG)==13: 
         continue
      # Counting neutrons
      elif (candPDG==2112 and candP4.P()>PNeutron): # Pandora FIXME: pion -> neutron misID 
         countNeutrons+=1
      # Counting pions and kaons
      elif candPDG==211 and candP4.P()>minP_pion:   # ignoring the difference between kaons and pions for now 
         countPions+=1
         # found_pions_id.add(key)
      # Counting photons (should be 2 x pi0s)
      elif candPDG==22 and candP4.P()>minP_photon:  # careful: here counting photons and not pi0s. Account for merged/lost photons.  
         countPhotons+=1
      else: 
         continue

      if maxConeTau<dR:
         maxConeTau=dR

      # Sum the charge and 4-momentum of the tau
      chargeTau+=cand.getCharge()
      tauP4+=candP4
      const[nConsts]=cand
      nConsts+=1

   #print (tauP4.Pt(),chargeTau,countPions,countPhotons)

   # set the ID: only valid combinations can be a tau (charge, constituents compatible with
   # tau decay). can be refined in the future. 
   if abs(chargeTau)==1 or not charge_condition:
      if (countPions==1 and countNeutrons==0):
         if countPhotons<10:
            tauID=countPhotons
         else:
            tauID=9
         # if countPhotons<=4:
            # tauID=countPhotons
         # if countPhotons>4:
               # tauID=5

      elif (countPions==3): 
         tauID = countPhotons+10 # 3 pions + photons
            # if countPhotons<=2:
            #    tauID=countPhotons+10 # more or less copied from the CMS convention for tauDecay
            # if countPhotons>2:
            #    tauID=countPhotons+10 # capping the number of photons

      elif (countPions==1 and countNeutrons>0): # Future FIXME: Pandora pion->neutron misID issue 
         # tauID=15
         tauID = -20 # To not interact with the other IDs 


      # return an object with P4, ID, Charge, AngleMax, nConsts, constIdx 
      # should be a class in the future
      # logger.debug(
      #    f"Tau con carga absoluta 1: "
      #    f"chargeTau: {chargeTau}, countPhotons: {countPhotons}, countPions: {countPions}, countNeutrons: {countNeutrons}, tauP4.P(): {tauP4.P()}, math.cos(tauP4.Theta()): {math.cos(tauP4.Theta())}, tauID: {tauID}"
      # )
      # print (tauP4.P(),math.cos(tauP4.Theta()),chargeTau,countPions,countPhotons,tauID)
      # if tauID == -1:
      #    logger.warning(
      #       f"Tau ID is -1, which is unexpected. "
      #       f"chargeTau: {chargeTau}, countPhotons: {countPhotons}, countPions: {countPions}, countNeutrons: {countNeutrons}"
      #    )
      return (tauP4,tauID,chargeTau,maxConeTau,nConsts,const), None

   else:
      # logger.debug(
      #    f"Tau con carga absoluta distinta de 1: "
      #    f"chargeTau: {chargeTau}, countPhotons: {countPhotons}, countPions: {countPions}, countNeutrons: {countNeutrons}"
      # )
      tauP4.SetXYZM(0,0,0,0) # safety, always return an object 
      return (tauP4,-1,0,0,0,dict()), None
                 

# loop over all gen taus 
def findAllGenTaus(mc_particles):
   """ Find all generator level taus.
   
   Args:
       mc_particles (Particle Collection): All particles in the event.
       
   Returns:
      genTaus (dict): Dictionary with the generator level taus containing tuples with the visible 4-momentum, the tau ID, and the charge.
   """
   genTaus={}
   nGenTaus=0
   for particle in mc_particles:
      # only taus
      if abs(particle.getPDG()) != 15:
         continue
      # in the pythia sample we need to check the genStatus:
      # (in some events we have several copies of the tau)
      
      # 2: final state tau (to not double count)
      if particle.getGeneratorStatus()!=2:
         continue
#        print ("genTau!",particle.getGeneratorStatus())

      # tauP4=ROOT.TLorentzVector()
      # tauP4.SetXYZM(particle.getMomentum().x,particle.getMomentum().y,particle.getMomentum().z,particle.getMass())

      genTau_data=visTauGen(particle)
      genTau = GenParticle(genTau_data[0], genTau_data[1], genTau_data[2], genTau_data[3], genTau_data[4], genTau_data[5], genTau_data[6])
      if genTau.getCharge()<0:
         genTau.setPDG(15)
      else:
         genTau.setPDG(-15)
      # visTauP4=genTau[0]
      # genTauId=genTau[1]

      genTaus[nGenTaus]=genTau
      nGenTaus+=1

   return genTaus

# function to find all reco taus starting from PFO collection 
def findAllTaus(pfos, dRMax, minP_photon, minP_pion, PNeutron, genminP, charge_condition=True):
   """ Find all tau candidates starting from PFO collection by recognizing the decay products.

   Args:
      pfos (PandoraPFOs): PandoraPFOs collection
      dRMax (float): Maximum cone radius.
      minP_photon (float): Minimum photon momentum.
      minP_pion (float): Minimum pion momentum.
      PNeutron (float): Minimum neutron momentum.
      genminP (float): Minimum general level momentum.

   Returns:
       taus (dict): Dictionary with the tau candidates containing tuples with the visible 4-momentum, the tau ID, and the charge.
   """
   taus={}
   nTaus=0
   # Dict to avoid duplicates
   # id_pfos = {i: pfos[i] for i in range(len(pfos))}
   found_pions_id = set()
   # for key, pf in id_pfos.items():
   for pf in pfos:
      if (abs(pf.getPDG())!=211): 
         continue 
      # if key in found_pions_id:
      #    continue
      pionP4 = ROOT.TLorentzVector()
      try:
         pionP4.SetXYZM(pf.getMomentum().x,pf.getMomentum().y,pf.getMomentum().z,pf.getMass())
      except AttributeError as e:
         pionP4.SetXYZM(pf.getMomentum().X(),pf.getMomentum().Y(),pf.getMomentum().Z(),pf.getMass())
         
      if pionP4.P() < minP_pion or  pionP4.P() < genminP:
         continue

      recoTau_data, pions_id = buildTauFromPion(pf, pfos, dRMax, minP_photon, minP_pion, PNeutron, genminP, charge_condition)
      recoTau = RecoParticle(recoTau_data[0], recoTau_data[1], recoTau_data[2], recoTau_data[3], recoTau_data[4], recoTau_data[5])
      # logger.debug(
      #    f"Id del RecoTau {recoTau.getID()}"
      # )
      if recoTau.getCharge()<0:
         recoTau.setPDG(15)
      else:
         recoTau.setPDG(-15)
      
      # # Add the main pion to pions_id
      # if pions_id:
      #    pions_id.add(key)
      #    found_pions_id.update(pions_id)
      # else:
      #    found_pions_id.add(key)
      
      
      candTauP4=recoTau.getMomentum()
      # candTauId=recoTau[1]
      # candTauCharge=recoTau[2]

      # # FIXME: this is very ugly, angular separation between taus to avoid duplicates
      # # in these samples most of the events have 1 tau (and 1 prong)
      # # in a real scenario this could be slow and we could veto events 
      duplicate=False
      for i in range(0,nTaus):
         if (myutils.dRAngle(candTauP4,taus[i].getMomentum())<0.05): duplicate=True
      if (duplicate==True): continue

      taus[nTaus]=recoTau
      nTaus+=1
      #print ("...",pf.getObjectID().index,candTauP4.Pt(),candTauP4.Phi(),candTauP4.Theta(),candTauId,candTauCharge)

   return taus

