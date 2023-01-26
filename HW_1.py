# -*- coding: utf-8 -*-
import time
import matplotlib.pylab as plt
import numpy as np
import scipy

#The energy should converge towards -2.855160

start = time.time()

alpha = [0.297104, 1.236745, 5.749982, 38.216677]

C = [1,1,1,1]




def generate_S(alpha):
    S = np.zeros((4,4))
    for p in range(len(alpha)):
        for q in range(len(alpha)):
            S[p][q] = (np.pi/(alpha[p]+alpha[q]))**0.5
    return S
S_matrix = generate_S(alpha)

def normalize(_C):
    out = _C/(np.sqrt( np.matmul(_C, np.matmul(S_matrix,_C)) ))
    return out

def generate_Q(alpha):
    Q = np.zeros((4,4,4,4))
    pi_term = 2*np.pi**(5/2)
    
    for p in range(len(alpha)):
        for r in range(len(alpha)):
            for q in range(len(alpha)):
                for s in range(len(alpha)):
                    Q[p][r][q][s] = (pi_term)/((alpha[p]+alpha[q])* \
                    (alpha[r]+alpha[s])*np.sqrt(alpha[p] + alpha[q] + \
                    alpha[r] + alpha[s]))

    return Q

Q_matrix = generate_Q(alpha)

def generate_H(alpha):
    H = np.zeros((4,4))
    pi4 = 4*np.pi
    for p in range(len(alpha)):
        for q in range(len(alpha)):
            H[p][q] = pi4/(alpha[p]+alpha[q]) * \
                (3/4 * alpha[q] *(1-alpha[q]/(alpha[p]+alpha[q]) )* \
                 np.sqrt(np.pi/(alpha[p]+alpha[q]) ) -1  )
    return H

def generate_S(alpha):
    S = np.zeros((4,4))
    for p in range(len(alpha)):
        for q in range(len(alpha)):
            S[p][q] = (np.pi/(alpha[p] + alpha[q]))**1.5
    
    return S

def generate_F(h,Q,C):
    F = np.zeros((4,4))
    for p in range(4):
        for q in range(4):
            F[p, q] = h[p, q]+np.matmul(C, np.matmul(Q[p, :, q, :], C))

    
    return F

def generate_energy(C, h, Q):
    out = 2*np.matmul(C, np.matmul(h, C))+ \
    np.tensordot(np.tensordot(np.tensordot(np.tensordot(Q, C, axes=([0], [0])),\
    C, axes=([0], [0])), C, axes=([0], [0])), C, axes=([0], [0]))
    return out



def self_consistency_solver(iterations,threshhold, C,Q,h,S):
    #Tensorproduct function
    E = np.zeros(iterations)
    for i in range(iterations):
        E[i] = generate_energy(C,h,Q)
        if np.abs(E[i]-E[i-1]) < threshhold:
            break
        
        F = generate_F(h,Q,C)
        
        #The Generalized Eigenvalue problem: FC = E'SC
        
        #Eigenvalue & Eigenvectors
        (epsilon, V) = scipy.linalg.eig(F, S)
    
        #Locate lowest energy eigenvalue and assign it to C
        C = V[:, np.argmin(epsilon.real)]
        C = normalize(C)
        

    print("calculated ground state energy = " + str(generate_energy(C, h, Q)))
    print("C-Coefficients = " +str(C))
        
        
#Generate matrices
Q_matrix = generate_Q(alpha)
S_matrix = generate_S(alpha)
H_matrix = generate_H(alpha)
C_matrix = normalize(C)
F_matrix = generate_F(H_matrix,Q_matrix,C_matrix)


iterations = 10
threshhold = 1e-5

self_consistency_solver( \
iterations,threshhold, C_matrix, Q_matrix, H_matrix, S_matrix)



end = time.time()
total_time = end - start
print("\n Execution time: "+ str(total_time))
