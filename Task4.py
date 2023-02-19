#Task 4
from ase.build import molecule
from gpaw import PW,GPAW
from ase.vibrations import Vibrations
from ase.thermochemistry import IdealGasThermo
from ase.calculators.emt import EMT
from ase.optimize import QuasiNewton



moleculeList = ["CO", "O2"] 
cutoff = 450 
potentialEnergyList = []
roomtemp = 300 # 300k
atmosphere = 10**5 #1bar


for material in moleculeList: #For task 4

    atoms = molecule(material, cell=(12, 12, 12))
    atoms.center()
    
    
    calc = GPAW(xc='PBE',
                mode=PW(cutoff),
                kpts={'gamma': True})
    atoms.set_calculator(calc)
    print(f"---------------Energy for {material} =  {atoms.get_potential_energy()} eV ------------------")



