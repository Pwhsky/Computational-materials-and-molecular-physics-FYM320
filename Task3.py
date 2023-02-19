#Construct 3 (111)-surfaces for Au, Pt,Rh.

# k-parameter should be (4,4,1)
# 6 Å of vaccuum
from ase.build import bulk
from ase.optimize import BFGS
from ase.io import Trajectory
import numpy as np
from ase.build import fcc111
from ase.io import write
from gpaw import GPAW, PW
from ase.optimize import GPMin



lattice_constants = [4.179, 3.969, 3.839]
materialList      = ["Au","Pt", "Rh"]

k = (4,4,1)
cutoff = 450

for i in range(3):

  material = materialList[i]
  surface = fcc111(material, (3, 3, 3), a=lattice_constants[i], vacuum=6.0)
  calc    = calc = GPAW(xc='PBE',
            mode=PW(cutoff),
            kpts=k)
  surface.set_calculator(calc)
  dyn = GPMin(surface)
  dyn.run(fmax=0.01, steps=100)
  surface.set_calculator(calc)
  print(surface.get_potential_energy())

