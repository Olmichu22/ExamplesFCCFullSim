from modules import tauReco, electronReco, muonReco
import ROOT
import edm4hep


def extractTauDecays(gatr_results_path,
                     mlpf_results,
                     eventid,
                     pfos,
                     dRMax,
                     minPTauPhoton,
                     minPTauPion,
                     PNeutron,
                     generalPCut,
                     foton_config,
                     test_extremes,
                     test_pfo,
                     logger_process):
  if gatr_results_path is not None and not test_pfo:
    logger_process.debug("Using GATr results for event %d", eventid)
    particles = mlpf_results.get(eventid, {})
    charge_condition = False
  else:
    logger_process.debug("Using PandoraPFO results for event %d", eventid)
    particles = pfos
    charge_condition = True
    
        
  recoTau = tauReco.findAllTaus(particles,
                                dRMax,
                                minPTauPhoton,
                                minPTauPion,
                                PNeutron,
                                generalPCut,
                                charge_condition=charge_condition)
  recoElectrons = electronReco.findAllElectrons(particles, generalPCut)
  recoMuons = muonReco.findAllMuons(particles, generalPCut)
        
  # Check extremes of photon energy
  if test_extremes:
      if test_pfo or gatr_results_path is None:
        particles_w_extremes_max = [p.clone() for p in particles]
        particles_w_extremes_min = [p.clone() for p in particles]
      else:
        particles_w_extremes_max = [p.copy() for p in particles]
        particles_w_extremes_min = [p.copy() for p in particles]
      for ind, part in enumerate(particles):
          pdg = abs(part.getPDG())
      
          if pdg == 22:
            if test_pfo or gatr_results_path is None:
              # If using PandoraPFOs, need to build TLorentzVector
              p4 = ROOT.TLorentzVector()
              p4.SetXYZM(part.getMomentum().x, part.getMomentum().y, part.getMomentum().z, part.getMass())
            else:
              # Using GATr results, can use directly because part is a RecoParticle
              p4 = part.getMomentum()
              
            p4_max, p4_min = tauReco.electromagnetic_energy_error_p4_extremes(p4, foton_config)
            if test_pfo or gatr_results_path is None:
              # Update momentum in the cloned PandoraPFOs
              new_p_max = edm4hep.Vector3f()
              new_p_max.x = p4_max.X()
              new_p_max.y = p4_max.Y()
              new_p_max.z = p4_max.Z()
              particles_w_extremes_max[ind].setMomentum(new_p_max)
              
              new_p_min = edm4hep.Vector3f()
              new_p_min.x = p4_min.X()
              new_p_min.y = p4_min.Y()
              new_p_min.z = p4_min.Z()
              particles_w_extremes_min[ind].setMomentum(new_p_min)
              
            else:
              particles_w_extremes_max[ind].setMomentum(p4_max)
              particles_w_extremes_min[ind].setMomentum(p4_min)
      
            logger_process.debug(f"Momentum changes max: {p4_max.P()} min: {p4_min.P()}")
      recoTau_max = tauReco.findAllTaus(
          particles_w_extremes_max, dRMax, minPTauPhoton, minPTauPion, PNeutron, generalPCut, charge_condition=charge_condition
      )
      recoTau_min = tauReco.findAllTaus(
          particles_w_extremes_min, dRMax, minPTauPhoton, minPTauPion, PNeutron, generalPCut, charge_condition=charge_condition
      )
  else: 
      recoTau_max = None
      recoTau_min = None
  return recoTau, recoElectrons, recoMuons, recoTau_max, recoTau_min