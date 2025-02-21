import ROOT

class GenParticle:
  """ Particle class to store the information of a particle.
  Attributes:
      p4 (TLorentzVector): 4-momentum of the particle.
      ID (int): PDG ID of the particle.
      charge (int): Charge of the particle.
      maxAngle (float): Maximum angle between the constituents of the particle.
      nConst (int): Number of constituents of the particle.
      const (dict): Dictionary with the constituents of the particle.
  """
  
  def __init__(self, visP4=None, ID=-1, charge=0, genP4=None, maxAngleConsts=0, nConsts=0, const=None):
    """ Constructor of the Particle class."""
    self.visp4 = visP4 if visP4 is not None else ROOT.TLorentzVector(0, 0, 0, 0)
    self.ID = ID
    self.charge = charge
    self.p4 = genP4 if genP4 is not None else ROOT.TLorentzVector(0, 0, 0, 0)
    self.maxAngle = maxAngleConsts
    self.nConst = nConsts
    self.const = const if const is not None else {}
    
  def getID(self):
    return self.ID

  def getCharge(self):
    return self.charge
  
  def getMomentum(self):
    return self.p4
  
  def getvisMomentum(self):
    return self.visp4
  
  def getMass(self):
    return self.p4.M()
  
  def getVisMass(self):
    return self.visp4.M()
  
  def getDaughters(self):
    return self.const
  
  def getnConst(self):
    return self.nConst
  
  def getMaxAngle(self):
    return self.maxAngle
  
  def setID(self, ID):
    self.ID = ID
  
  def setCharge(self, charge):
    self.charge = charge
  
  def setMomentum(self, candP):
    self.p4.SetXYZM(candP.getMomentum().x,
                    candP.getMomentum().y,
                    candP.getMomentum().z,
                    candP.getMass())
  
  def setvisMomentum(self, visP4):
    self.visp4 = visP4
  
  def setDaughters(self, const):
    self.const = const
    self.nConst = len(const)
  
  def addDaughter(self, daughter):
    self.const[self.nConst] = daughter
    self.nConst += 1
  
  def setnConst(self, nConst):
    self.nConst = nConst
  
  def setMaxAngle(self, maxAngle):
    self.maxAngle = maxAngle
    
  def ShowInfo(self):
    """ Print the information of the particle and its constituents."""
    print("GenParticle ID: %d, Charge: %d, Mass: %f, VisMass: %f, nConst: %d, Angle: %d" % (self.ID, self.charge, self.p4.M(), self.visp4.M(), self.nConst, self.maxAngle))
    for i in range(self.nConst):
      print("  Constituent %d:" % i)
      print(self.const[i])
  
  def __str__(self):
    return "ID: %d, Charge: %d, Mass: %f, VisMass: %f, nConst: %d, Angle: %d" % (self.ID, self.charge, self.p4.M(), self.visp4.M(), self.nConst, self.maxAngle)