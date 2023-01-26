import numpy as np
from matplotlib import pyplot as plt
#Poissons eq. using FDM
alpha = np.array([0.297104, 1.236745, 5.749982, 38.216677])



N = 1000
a = float(0)
b = float(5)
h = (b-a)/N


rho = np.arange(1,N+1)*h
hydrogen_density = 1/np.pi*np.exp(-2*rho)



#Task 3
def generate_hydrogen_wavefunction_FDM(rho):
    #Coefficient matrix
    A = np.diag((h**-2 -rho**-1),0)+np.diag(-np.ones(N-1)/(2*h**2),1)+np.diag(-np.ones(N-1)/(2*h**2),-1)
    #Eigenvalue solution
    (E,X) = np.linalg.eig(A)
    epsilon = np.min(E).real
    
    
    temp = X[:,np.argmin(E)]
    temp = temp*(np.sqrt(np.trapz(temp**2,rho))*np.sign(temp[0])**-1)
    
    wavefunction = (np.sqrt(4*np.pi)*temp*rho**-1)
    
    #Normalize
    wavefunction = wavefunction/(np.trapz(wavefunction,rho))
    return (epsilon,wavefunction)


#Task 2
def generate_hartree_analytical(r):
    V_H = np.zeros(N)
    for i in range(0,N):
        V_H[i] = rho[i]**-1 - (1+ rho[i]**-1)*np.e**(-2*rho[i])
    return V_H


def generate_poisson_solution_FDM(rho,a,b):
    #A = coeff matrix for the discrete domain
    _A = np.diag(-2*np.ones(N),0)+np.diag(np.ones(N-1),1)+np.diag(np.ones(N-1),-1)
    temp_potential = -4*np.pi*h**2*rho*hydrogen_density
    temp_potential[-1] -= 1
    U_0 = np.linalg.solve(_A,temp_potential)
    U = U_0 * rho**-1
    return U
    
#Plot hartree vs Poisson
plt.figure()
plt.plot(rho,generate_hartree_analytical(rho),'red',label = r"Hartree potential")
plt.plot(rho,generate_poisson_solution_FDM(rho,a,b),'b--', label = r"FDM-Poisson")
plt.title("Poisson's equation solved with FDM")
plt.xlabel("r ")
plt.ylabel("U (r)")
plt.legend()


#Plot radial solution to S.E and the Hydrogen ground state
hydrogen_wavefunction_FDM = generate_hydrogen_wavefunction_FDM(rho)
epsilon = hydrogen_wavefunction_FDM[0]
psi = np.array(hydrogen_wavefunction_FDM[1])
print("Hydrogen ground state energy: " + str(epsilon) + "[HF]")
plt.figure()
plt.plot(rho, 4*np.pi*rho**2*np.abs(psi)**2)
plt.title("Radial S.E for hydrogen ground state " )
plt.xlabel("r ")
plt.ylabel("P (r)")



