import sys
import math
import ROOT
from array import array
from podio import root_io
import edm4hep
from modules.ParticleObjects import GenParticle, RecoParticle

from modules import myutils


def findAllMuons(pfos, minPt):
  """ Find all tau candidates starting from PFO collection by recognizing the decay products.

  Args:
      pfos (PandoraPFOs): PandoraPFOs collection
      minPt (float): Minimum particle momentum.
  Returns:
      muons (dict): Dictionary with the muon candidates containing RecoParticle Objects.
  """
  muons={}
  nMuons=0
  for pf in pfos:
    if (abs(pf.getPDG())!=13): 
        continue 

    
    muonP4 = ROOT.TLorentzVector()
    muonP4.SetXYZM(pf.getMomentum().x,pf.getMomentum().y,pf.getMomentum().z,pf.getMass())
    if muonP4.P()<minPt:
        continue
    muonpdg = pf.getPDG()
    muoncharge = pf.getCharge()
    
    muon = RecoParticle(p4 = muonP4, ID = -1, charge = muoncharge, PDGID=muonpdg)      

    muons[nMuons]=muon
    nMuons+=1
    #print ("...",pf.getObjectID().index,candTauP4.Pt(),candTauP4.Phi(),candTauP4.Theta(),candTauId,candTauCharge)

  return muons