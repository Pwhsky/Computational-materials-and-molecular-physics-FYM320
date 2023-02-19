from ase.build import bulk
from ase.optimize import BFGS
from ase.io import Trajectory
from gpaw import GPAW, PW
from ase.constraints import StrainFilter
import numpy as np

#Code for task 1 and 2


latticeConstantGuesses = [4.16, 3.93, 3.81]
materialList           = ["Au","Pt", "Rh"]
DFT_referenceEnergies  = [-0.158, -0.704, -1.218 ] #This is needed for task 2!


#Parameters:
k = 12 #resolution
cutoff = 450 
step_size = 0.01


energies=[]
for i in range(3):  #Material loop
    
    latticeConstantGuess = latticeConstantGuesses[i]
    material = materialList[i]
    print(f"material: {material}  --------------")


    #Generate a list of lattice constants to examine for a given material
    a_trials = np.arange(latticeConstantGuess-4*step_size,
                         latticeConstantGuess+8*step_size,
                         step=step_size)   #Length 12 or 13
     

    for a_trial in a_trials:  #Loop through a list of trial lattice constants

       #structure = bulk(material,"fcc", cubic = True, a = a_trial  ) # <- this gives shit energies

       b = a_trial/2
       structure = Atoms(material,pbc = True, 
                    cell = [ [0, b, b],
                             [b, 0, b],
                             [b, b, 0] ]) 
       
#########     el calculator     #######
       calculator = GPAW(mode=PW(cutoff),
                      xc='PBE',
                      txt=f'{material}.out',
                      kpts=(k, k, k),
                      communicator=None)
       structure.calc = calculator
#########################################       

       potential_energy = structure.get_potential_energy()

       #Append the energy along with the trialed lattice constant
 
       energies.append(potential_energy)
       print(f"a = {a_trial}")
       print(f"E = {potential_energy}eV")
       print("---------------------------")
#################################################


#tutorial: https://wiki.fysik.dtu.dk/ase/tutorials/lattice_constant.html