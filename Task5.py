#Task 5
from ase.build import molecule
from gpaw import PW,GPAW
from ase.vibrations import Vibrations
from ase.thermochemistry import IdealGasThermo
from ase.calculators.emt import EMT
from ase.optimize import QuasiNewton
from scipy import constants
import numpy as np


entropyList = np.zeros(2)
moleculeList = ["CO", "O2"] 
 #parameters for thermoCalculator (see https://wiki.fysik.dtu.dk/ase/ase/thermochemistry/thermochemistry.html#module-ase.thermochemistry)
spin_types = [0,1]
symmetries = [1,2]

cutoff = 450 
potentialEnergyList = []
roomtemp = 300 # 300k
atmosphere = 10**5 #1bar


for i in range(len(moleculeList)): #For task 5

    material = moleculeList[i]
    spin = spin_types[i]
    symmetry = symmetries[i]


    atoms = molecule(material, cell=(12, 12, 12))
    atoms.calc = EMT()
    dyn = QuasiNewton(atoms)
    dyn.run(fmax=0.01)
    potentialenergy = atoms.get_potential_energy()
    
    vib = Vibrations(atoms)
    vib.run()
    vib_energies = vib.get_energies()
    
    thermo = IdealGasThermo(vib_energies=vib_energies,
                        potentialenergy=potentialenergy,
                        atoms=atoms,
                        geometry='linear',
                        symmetrynumber=symmetry, spin=spin) 
    
    S = thermo.get_entropy(temperature=roomtemp, pressure=atmosphere) 
    entropyList[i] = S

#convert to j/K
eVToJoule = 1.602176565e-19
entropyList = entropyList*eVToJoule*(constants.Avogadro)

print(f"Entropy for CO and O2 respectively = "{entropyList})

