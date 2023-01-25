
import numpy as np
import scipy
from matplotlib import pyplot as plt
#Task 5
Z=2
N = 1000
a = float(0)
b = float(5)
h = (b-a)/N
rho = np.arange(1,N+1)*h


initial_guess =1/np.pi*Z**3*np.exp(-2*Z*rho) 

def generate_wavefunction(rho,V,Z):
    #Discrete Laplace operator
    diagonal_term = h**-2 - Z*rho**-1 + V
    A = np.diag(diagonal_term,0)+np.diag(-np.ones(N-1)/(2*h**2),1)+np.diag(-np.ones(N-1)/(2*h**2),-1)
    #Eigenvalues & eigenvectors:
    (Eigenvalues,Eigenvectors) = np.linalg.eig(A)
  
    # X contains N eigenvalues, build the wavefunction
    #with the lowest eigenvalue:
    
    minimized_energy_vector = Eigenvectors[:,np.argmin(Eigenvalues)]
    temp = minimized_energy_vector
    temp = temp*(np.sqrt(np.trapz(temp**2,rho))*np.sign(temp[0])**-1)
    
    wavefunction = (np.sqrt(4*np.pi)**-1)*temp*rho**-1
    
    #Normalize
    #wavefunction = wavefunction/(np.trapz(wavefunction,rho))
    epsilon = np.min(Eigenvalues).real
    
    return (epsilon,wavefunction)


def generate_energy(epsilon,epsilon_xc, V_H, V_xc, density, Z):
    E = Z*epsilon-4*Z*\
        np.pi*np.trapz((V_H*density/2+V_xc*density-epsilon_xc*density)*rho**2,rho)  
    return E

def generate_exchange_terms(Z,density):
    epsilon_xc = -3/4*(3*Z*density/np.pi)**(1/3)
    potential_xc = -1*(3*Z*density/np.pi)**(1/3)
    return epsilon_xc, potential_xc

def generate_hartree(rho,N,density):
    A = np.diag(-2*np.ones(N),0)+np.diag(np.ones(N-1),1) + np.diag(np.ones(N-1),-1)
    potential = -4*np.pi*h**2*rho*density
    potential[-1] -= 1
    V_H = (np.linalg.solve(A,potential))*rho**-1
    return V_H

def self_consistency_solver(iterations,threshhold):
    E = np.zeros(iterations)
    density = initial_guess
    epsilon_xc,V_xc = generate_exchange_terms(Z,density)
    V_H = Z*generate_hartree(rho,N,density)
    for i in range(iterations):
       
        
        psi = generate_wavefunction(rho,V_xc +V_H,Z)[1]
        epsilon = generate_wavefunction(rho,V_xc +V_H,Z)[0]
        
        density = np.abs(psi)**2
        epsilon_xc,V_xc = generate_exchange_terms(Z,Z*density)
        V_H = Z*generate_hartree(rho,N,density)
        E[i] = generate_energy(epsilon,epsilon_xc,V_H,V_xc, density,Z)
        if np.abs(E[i]-E[i-1]) <threshhold:
            print("E_g = " + str( E[i]))
            break
        
    
    
    print("E = " + str(E[i]))


iterations = 50
threshhold = 1e-5/27.21
self_consistency_solver(iterations,threshhold)
