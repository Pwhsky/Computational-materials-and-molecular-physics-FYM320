
import numpy as np
import scipy
from matplotlib import pyplot as plt
#Task 5
Z=2
N = 1000
a = float(0)
b = float(5) #This can be reffered to as rmax in some cases
h = (b-a)/N

rho = np.arange(1,N+1)*h
initial_guess =1/np.pi*Z**3*np.exp(-2*Z*rho) 

A, B, C, D, gamma, beta_1, beta_2 = 0.0311, -0.048, 0.0020, -0.0116, -0.1423, 1.0529, 0.3334


def generate_rs(density,Z):
    return (3/(4*np.pi*Z*density))**(1/3)




def generate_wavefunction(rho,V,Z):
    #Returns a wave function and epsilon energy
    diagonal_term = h**-2 - Z*rho**-1 + V
    _A = np.diag(diagonal_term,0)+np.diag(-np.ones(N-1)/(2*h**2),1)+np.diag(-np.ones(N-1)/(2*h**2),-1)
    #Eigenvalues & eigenvectors:
    (Eigenvalues,Eigenvectors) = np.linalg.eig(_A)
  
    # X contains N eigenvalues, build the wavefunction
    #with the lowest eigenvalue:
    
    minimized_energy_vector = Eigenvectors[:,np.argmin(Eigenvalues)]
    temp = minimized_energy_vector
    temp = temp/(np.sqrt(np.trapz(temp**2,rho))*np.sign(temp[0]))
    
    wavefunction = 1/(np.sqrt(4*np.pi))*temp*(rho**-1)
    
    #Normalize?
    #wavefunction = wavefunction/(np.trapz(wavefunction,rho))
    epsilon = np.min(Eigenvalues).real
    
    return (epsilon,wavefunction)


def generate_energy(epsilon,epsilon_xc, V_H, V_xc, density, Z):
    E = Z*epsilon-4*Z*\
        np.pi*np.trapz((V_H*density/2+V_xc*density-epsilon_xc*density)*rho**2,rho)  
    return E


def generate_correlation_terms(density,Z):
    r=generate_rs(density,Z)
    #Correlation energies for the two domains r >= 1 & r<1
    epsilon_correlation=(r<1)*(A*np.log(r)+B+C*r*np.log(r)+D*r)
    epsilon_correlation=epsilon_correlation+ (r>=1)*gamma/(1+beta_1*np.sqrt(r)+beta_2*r)
    
    
    #Correlation potential
    V_c=(r>=1)*epsilon_correlation*(1+7/6*beta_1*np.sqrt(r)+4/3*beta_2*r)/(1+beta_1*np.sqrt(r)+beta_2*r)
    V_c=(r<1)*(A*np.log(r)+B-A/3+2/3*C*r*np.log(r)+(2*D-C)*r/3)
    
    return epsilon_correlation, V_c

def generate_exchange_terms(Z,density):
    epsilon_xc = -3/4*(3*Z*density/np.pi)**(1/3)
    potential_xc = -1*(3*Z*density/np.pi)**(1/3)
    return epsilon_xc, potential_xc


def generate_hartree(rho,N,density):
    _A = np.diag(-2*np.ones(N),0)+np.diag(np.ones(N-1),1) + np.diag(np.ones(N-1),-1)
    potential = -4*np.pi*h**2*rho*density
    potential[-1] -= 1.5
    V_H = (np.linalg.solve(_A,potential))*(rho**-1)
    return V_H




def self_consistency_solver(iterations,threshhold):
    E = np.zeros(iterations)
    density = initial_guess
    
    epsilon_xc,V_xc = generate_exchange_terms(Z,Z*density)
    epsilon_c, V_c = generate_correlation_terms(Z*density,Z)
    
    
    epsilon_sum = epsilon_xc + epsilon_c
    V_sum = V_xc+ V_c
    for i in range(iterations):
        V_H = Z*generate_hartree(rho,N,density)
        psi = generate_wavefunction(rho,V_sum +V_H,Z)[1]
        epsilon = generate_wavefunction(rho,V_sum +V_H,Z)[0]
        
        density = np.abs(psi)**2
        
        epsilon_xc,V_xc = generate_exchange_terms(Z,Z*density)
        epsilon_c, V_c = generate_correlation_terms(Z*density,Z)
        
        epsilon_sum = epsilon_xc + epsilon_c
        V_sum = V_xc+ V_c
       # V_H = Z*generate_hartree(rho,N,density)
        E[i] = generate_energy(epsilon,epsilon_sum,V_H,V_sum, density,Z)
        if np.abs(E[i]-E[i-1]) <threshhold:
            print("E_g = " + str( E[i]))
            break
        
    
    
    print("E = " + str(E[i]))


iterations = 50
threshhold = 1e-5
self_consistency_solver(iterations,threshhold)