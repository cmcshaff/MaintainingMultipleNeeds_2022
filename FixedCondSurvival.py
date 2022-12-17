# Parametric Scan of [N] and [F] 

# Connor McShaffrey; 7/6/22

# SUPPLEMENRARY CODE FOR THE PAPER "MAINTAINING VIABILITY WITH MULTIPLE NEEDS" (MCSHAFFREY & BEER, 2022)

## This code is for the parametric scan of [N] and [F], the food molecules of the protocell.
## The purpose of this is to see which fixed combinations of [N] and [F] are able to support
## the metabolic dynamics of the protocell given different starting conditions. 

##############################################################################

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import multiprocessing as multip
from scipy.integrate import odeint 

##############################################################################

'The ODEs in a Single Function'

# This ODE is for the metabolic dynamics given a fixed combination of [N] and 
# [F] without behavior. 

def ode_func(vec,t,N,F):
    # Constants
    N_conc = N
    F_conc = F
    gamma = 0.04
    kd = 0.05
    K1 = 4
    n = 3
    
    
    # ODE vector element
    A = vec[0]
    B = vec[1]
    
    # Coupled Equations
    dAdt = gamma * ((B**n)/((K1**n) + (B**n))) * N_conc - kd*A
    dBdt = gamma * ((A**n)/((K1**n) + (A**n))) * F_conc - kd*B
    
    #Returning Each
    return(dAdt,dBdt)
    
##############################################################################

'Single Run'

# This function takes N and F parameters and initial conditions for A and B,
# and returns whether the system survives or not. 

def AB_survival(N,F,a,b):
    
    survival = 0.0
            
    x0 = [a,b] # Initial conditions

    #t = np.linspace(0,5,20) # Duration and resolution of the simulation
    t = np.linspace(0,800,4000)

    x = odeint(ode_func, x0, t, args=(N,F))
    

    A = x[:,0]
    B = x[:,1]

    
    C = A+B
    
    if (all(a > 0.1 for a in A) and all(b > 0.1 for b in B) and all(c <= 20 for c in C)):
        survival += 1.0
                                        
    return(survival)

##############################################################################

'Helper Function'

# This function makes it so that all the various inputs for the single run 
# become one tuple so that we can use parallel processing with it. 

def helper(z):
    N, F, A, B = z
    survival = AB_survival(N, F, A, B)
    return (survival)

##############################################################################

'[A] & [B] Assessment'

# Assessing combinations of [A] and [B] in a range of [0,20] for a given 
# concentration of [N] and [F]. We are looking for 50x50=2500 combinations
# of [A] and [B]. This code is parallelized. 

def AB_assess(N, F):
    
    # Initializing A's and B's that need to be scanned over.
    init_A = np.linspace(0,20,50)
    init_B = np.linspace(0,20,50)
    conds = []
    survivability = 0.0 # Normalized score for survivability.
    
    # Making a series of tuple inputs for the helper function.
    for a in init_A:
        for b in init_B:
            conds.append((N,F,a,b))
            
    # Attempting parallel processing.
    if __name__ == '__main__':
        num_workers = multip.cpu_count()
        pool = multip.Pool(num_workers)
        #pool = multip.Pool(processes=4)
        outputs = pool.map(helper, conds)
        #print("Input: {}".format(inputs))
        #print("Output: {}".format(outputs))
            
        survivability = np.mean(outputs)
    
    return(survivability)

##############################################################################

'Scanning Environmental Conditions as Parameters'

# This function is for scanning across some resolution of N and F combinations
# treated as parameters [0,50]. Note that since the system is symmetric, a more
# efficient way to do this is to only scan over half of the matrix and then
# reflect it over the diagonal.

# This is a version of the code that does not use the diagonal trick. 

def env_combs(resolution): # Specify how many combinations to look at
    mapping = np.zeros([resolution,resolution]) # Create a matrix to store
                                                # them in.
    
    
    N_array = np.linspace(0,50,resolution) # [N]'s to scan
    F_array = np.linspace(0,50,resolution) # [F]'s to scan
    F_array = np.flip(F_array) # Flipping array for the purposes of plotting
                               # with values increasing along the y-axis.
    
    for el_n, n in enumerate(N_array):
        for el_f, f in enumerate(F_array):
            mapping[el_f,el_n] = AB_assess(n,f)
        
        
    return(mapping)
            
            
##############################################################################

'Looking at Survival'

# This is the code for looking at the survivability of the system as a function
# of [N] and [F], which are treated as parameters (ie fixed environment).

scan = env_combs(100)

sns.heatmap(scan)

# im = plt.imshow(data['nf_scan'], cmap='hot')
# plt.colorbar(im)
# plt.title('[N] & [F] Parameter Scan')
# plt.xlabel('[N]')
# plt.ylabel('[F]')
# plt.xticks([0,20,40,60,80,100], [0,10,20,30,40,50])
# plt.yticks([0,20,40,60,80,100], [50,40,30,20,10,0])
# plt.rcParams["figure.dpi"] = 1000
# plt.show()

# im = plt.imshow(param['nf_scan'], cmap='hot')
# plt.colorbar(im)
# plt.title('Parameter Space ([N] & [F])')
# plt.xlabel('[N]')
# plt.ylabel('[F]')
# plt.plot([0,99],[99,0], color='green')
# plt.plot([0,0],[0,61.38], color='violet')
# plt.plot([37.62,99],[99,99], color='violet')
# plt.plot([0,37.62],[61.38,99], color='violet')
# plt.xticks([0,20,40,60,80,100], [0,10,20,30,40,50])
# plt.yticks([0,20,40,60,80,100], [50,40,30,20,10,0])
# plt.rcParams["figure.dpi"] = 1000
# plt.show()

