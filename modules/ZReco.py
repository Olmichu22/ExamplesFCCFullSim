import sys
import math
import ROOT
from array import array
from podio import root_io
import edm4hep
from modules import tauReco, muonReco, electronReco
from modules import myutils
from modules.ParticleObjects import GenParticle, RecoParticle
    
  

# Check a generator level tau candidate, find the decay, 
# and compute visible (meson) variables 
def visTauZ(candZ):
  """ Check a generator level Z candidate, find the decay, and compute visible (meson) variables.

  Args:
       candZ (Particle Object): Particle object with the tau candidate information.

  Returns:
       Tuple: Tuple with the visible 4-momentum, the tau ID, the charge, the true 4-momentum, the maximum angle between constituents, the number of constituents, and the constituents.
  """
  genZ = GenParticle()
  
  # ID=1 for Z->tau tau decay
  genZ.setID(1)
  genZ.setCharge(candZ.getCharge())
  genZ.setMomentum(candZ)
  
  nconst = 0
  const = {}
  visZP4 = ROOT.TLorentzVector(0, 0, 0, 0)
  chargeZ = 0
  
  # loop over daughter particles of the Z
  for dau in candZ.getDaughters():
    visGenTau = tauReco.visTauGen(dau)
    visGentauParticle = GenParticle(*visGenTau)
    const[nconst] = visGentauParticle
    nconst += 1
    visZP4 += visGentauParticle.getvisMomentum()
    # print(f"Mass dau {nconst}: {visGentauParticle.getVisMass()}")
    # print(f"Cummulate Mass {visZP4.M()}")
    chargeZ += visGentauParticle.getCharge()

  # set the maximum angle between the
  # constituents
  # only works for Z->tau tau
  if nconst == 2:
    daughters_angle = myutils.dRAngle(const[0].getMomentum(),
                                        const[1].getMomentum())
    genZ.setMaxAngle(daughters_angle)

  
  # automatically set nconst  
  genZ.setDaughters(const)
  # set visible 4-momentum
  genZ.setvisMomentum(visZP4)
  genZ.setCharge(chargeZ)
   
  return genZ

def findZ(pfos, dRMax, minP, PNeutron):
  """ Find Z candidate starting from PFO collection by recognizing the decay products.
  
  Args:
      pfos (PandoraPFOs): PandoraPFOs collection
      dRMax (float): Maximum distance between the tau and the PFO.
      minP (float): Minimum particle momentum.
      PNeutron (float): Neutron momentum.
  Returns:
      recoZ (dict): Z candidate containing RecoParticle Objects."""
  recoTaus= tauReco.findAllTaus(pfos,dRMax, minP,PNeutron)
  recoMuons = muonReco.findAllMuons(pfos, minP)
  recoElectrons = electronReco.findAllElectrons(pfos, minP)
  nTaus=len(recoTaus)
  nMuons=len(recoMuons)
  nElectrons=len(recoElectrons)
  
  nLeptons=nMuons+nElectrons+nTaus
  if nLeptons==0 or nLeptons != 2:
    return None
  
  recoLeptons = {}
  eventype_id_dict = {"muonmuon": 0,
                      "muonelectron": 1,
                      "muontau": 2,
                      "electrontau": 3,
                      "tautau": 4}
  key = ""
  for i, muon in recoMuons.items():
    recoLeptons[f"muon{i}"] = muon
    key += "muon"
  for i, electron in recoElectrons.items():
    recoLeptons[f"electron{i}"] = electron
    key += "electron"
  for i, tau in recoTaus.items():
    # Ignore muons and electrons when looking for taus
    if tau.getID() == -13 or tau.getID() == -11:
      continue
    recoLeptons[f"tau{i}"] = tau
    key += "tau"
  
  tot_charge = 0
  zP4 = ROOT.TLorentzVector() 
  for i, lepton in recoLeptons.items():
    tot_charge += lepton.getCharge()
    zP4 += lepton.getP4()
  
  if tot_charge != 0:
    return None
  
  even_type_id = eventype_id_dict[key]
  recoZ = RecoParticle(zP4, even_type_id, 0, 0, 2, recoLeptons, 23)
  return recoZ
  
  
# loop over all gen taus 
def findAllGenZs(mc_particles):
  """ Find all generator level Zs.
  
  Args:
      mc_particles (Particle Collection): All particles in the event.
      
  Returns:
    genZa (dict): Dictionary with the generator level Zs containing a GenParticle Object.
  """
  genZs={}
  nGenZ=0
  for particle in mc_particles:
    # only Z
    if abs(particle.getPDG()) != 23:
        continue
    # in the pythia sample we need to check the genStatus:
    # (in some events we have several copies of the tau)
    
    # 2: final state tau (to not double count)
    
    # print("StatusID: ", particle.getGeneratorStatus())
    # print("Hijos: ", [Pid.getPDG() for Pid in particle.getDaughters()])
    # print("Status hijos: ", [Pid.getGeneratorStatus() for Pid in particle.getDaughters()])
    # print("\n")
    
    # Tau decay and final state in the daughters
    if 15 not in [Pid.getPDG() for Pid in particle.getDaughters()] or 2 not in [Pid.getGeneratorStatus() for Pid in particle.getDaughters()]:
        continue
    

    # ZP4=ROOT.TLorentzVector()
    # ZP4.SetXYZM(particle.getMomentum().x,particle.getMomentum().y,particle.getMomentum().z,particle.getMass())

    genZ=visTauZ(particle)
    # visZP4=genZ[0]
    # genZId=genZ[1]

    genZs[nGenZ]=genZ
    nGenZ+=1

  return genZs


def findOneZ(mc_particles):
  """ Find all generator level taus.
  
  Args:
      mc_particles (Particle Collection): All particles in the event.
      
  Returns:
    genTaus (dict): Dictionary with the generator level taus containing tuples with the visible 4-momentum, the tau ID, and the charge.
  """
  ZFound = False
  genZ = None
  for particle in mc_particles:
    # only Z
    if abs(particle.getPDG()) != 23:
        continue
    
    # Tau decay and final state in the daughters
    if 15 not in [Pid.getPDG() for Pid in particle.getDaughters()] or 2 not in [Pid.getGeneratorStatus() for Pid in particle.getDaughters()]:
        continue
    

    # ZP4=ROOT.TLorentzVector()
    # ZP4.SetXYZM(particle.getMomentum().x,particle.getMomentum().y,particle.getMomentum().z,particle.getMass())
    print("Found Z event! \n")
    ZFound = True
    genZ = visTauZ(particle)
    
    genZ.ShowInfo()
    
  return genZ, ZFound 
    # visZP4=genZ[0]
    # genZId=genZ[1]