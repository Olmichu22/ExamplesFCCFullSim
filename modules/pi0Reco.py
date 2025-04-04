import sys
import math
import ROOT
from array import array
from podio import root_io
import edm4hep
from modules.ParticleObjects import GenParticle, RecoParticle
from modules import myutils
import itertools

import logging
logger = logging.getLogger("pi0mass")

PI0INVARIANTMASS = 0.1349768 # GeV

def getPi0Mass(photons, strategy):
  
  
  # Conditions to not calculate the mass
  if len(photons) < 2:
    # Return non matched photon
    return None, photons
  elif len(photons)==2:
    # Only two photons, return mass
    P1 = ROOT.TLorentzVector()
    P2 = ROOT.TLorentzVector()
    P1.SetXYZM(
                photons[0].getMomentum().x,
                photons[0].getMomentum().y,
                photons[0].getMomentum().z,
                photons[0].getMass(),
            )
    P2.SetXYZM(
                photons[1].getMomentum().x,
                photons[1].getMomentum().y,
                photons[1].getMomentum().z,
                photons[1].getMass(),
            )
    return cumulatedPhotonsMass(P1, P2), None
  
  # Posible pairs combinations
  combinations = itertools.combinations(photons.keys(), 2)

  photon_momentums = {}
  for i in photons.keys():
    P = ROOT.TLorentzVector()
    P.SetXYZM(
                photons[i].getMomentum().x,
                photons[i].getMomentum().y,
                photons[i].getMomentum().z,
                photons[i].getMass(),
            )
    photon_momentums[i] = P
  logger.debug(f"Trying to find best pair of photons with strategy {strategy} {list(strategy.keys())[0]}")
  # Two strategies: mass and distance
  if list(strategy.keys())[0] == "mass":
    # Minimum acceptable mass
    min_mass = strategy.get("mass", -1)
    
    best_mass = 99999999
    best_pair = None
    for i,j in list(combinations):
      mass = cumulatedPhotonsMass(photon_momentums[i], photon_momentums[j])
      
      logger.debug(f"Mass of pair {i} and {j} is {mass}")
      
      # Evaluate new pair mass
      if abs(best_mass-PI0INVARIANTMASS) > abs(mass-PI0INVARIANTMASS):
        
        logger.debug(f"Changing best mass {best_mass} to {mass}")
        
        best_mass = mass
        best_pair = [i,j]
        
    
    # Evaluate result    
    if best_pair is not None and best_mass > min_mass:
      # Get non matched photons
      non_matched_photons = [k for k in photons.keys() if k not in best_pair]
      non_matched_photons = {k:photons[k] for k in non_matched_photons}
      logger.debug(f"Non matched photons are {non_matched_photons.keys()}")
      return best_mass, non_matched_photons
    
    else:
      logger.warning(f"No best pair found. Best pair is {best_pair} with mass {best_mass}")
      return None, photons
        
  elif list(strategy.keys())[0] == "distance":
    if strategy["distance"] != -1:
      max_distance = strategy["distance"]
    else:
      max_distance = 99999999
    
    min_distance_pair = None
    min_distance_value = 99999999
    for i,j in list(combinations):
      distance = myutils.dRAngle(photon_momentums[i], photon_momentums[j])
      
      logger.debug(f"Distance of pair {i} and {j} is {distance}")
      
      # Evaluate new pair mass
      if abs(min_distance_value) > abs(distance):
        
        logger.debug(f"Changing best distance {min_distance_value} to {distance}")
        
        min_distance_value = distance
        min_distance_pair = [i,j]
      
    if min_distance_pair is not None and min_distance_value < max_distance:
      logger.debug(f"Best distance pair is {min_distance_pair}")
      
      non_matched_photons = [k for k in photons.keys() if k not in min_distance_pair]
      non_matched_photons = {k:photons[k] for k in non_matched_photons}

      logger.debug(f"Non matched photons are {non_matched_photons}")
      
      mass = cumulatedPhotonsMass(photon_momentums[min_distance_pair[0]], photon_momentums[min_distance_pair[1]])
      
      return mass, non_matched_photons
    else:
      logger.warning(f"No best pair found. Best pair is {min_distance_pair} with distance {min_distance_value}")
      return None, photons
    
  else:
    logger.warning("Unknown strategy for pi0 mass calculation")
    return None, None
  

def cumulatedPhotonsMass(P1, P2):
  
  P1 += P2
  return P1.M()