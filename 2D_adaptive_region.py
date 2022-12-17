# COUNTERBALANCE VIABILITY BOUNDARY (2D)

## Connor McShaffrey; 9/14/22

## SUPPLEMENRARY CODE FOR THE PAPER "MAINTAINING VIABILITY WITH MULTIPLE NEEDS" (MCSHAFFREY & BEER, 2022)

### This is a version of the code that I used to generate the 2D slices of the
### regions of survivable initial conditions, but formatted for parallel
### processing. This helps because numerically integrating from so many initial 
### conditions is computationally expensive. The code can be easily changed to
### run normally. 

### In this case, we are taking a slice at a specific location, 'L', and 
### assuming that [X] = 0.5, which means that the protocell starts off not
### moving in any direction. 

### With the benefit of hindsight, there are ways in which this code could be improved,
### but it works well enough for the purposes of this paper. 

##############################################################################

'Libraries'

import numpy as np
import matplotlib.pyplot as plt
import multiprocessing as multip
from scipy.integrate import odeint 


##############################################################################

'Conditionals for Equations'

# These functions are called in the equations to make sure that the chemical gradient
# never jumps to a negative value. Hence we have a linear slope that is zero.  

# Amount of 'N' available based on location
def N(L,S,C):
    if S*L+C < 0.0:
        N = 0.0
    else:
        N = S*L+C
    return(N)
      
# Amount of 'F' available based on location  
def F(L,S,C):
    if -S*L+C < 0.0:
        F = 0.0
    else:
        F = -S*L+C
    return(F)

##############################################################################

'The ODEs in a Single Function'

# This is the ODE for the behaving protocell and returns vectors of the various
# states that have occurred during the time evolution of the system.

def ode_func(vec,t):
    # Constants
    alpha = 1
    gamma = 0.04
    kd = 0.05
    K1 = 4
    n = 3
    h = 20
    K2 = 0.5
    C = 9
    S = 4
    V = 2
    
    # ODE vector element
    L = vec[0]
    A = vec[1]
    B = vec[2]
    X = vec[3]
    
    # Coupled Equations
    dLdt = (X - 0.5)*V
    dAdt = gamma * ((B**n)/((K1**n) + (B**n))) * N(L,S,C) - kd*A
    dBdt = gamma * ((A**n)/((K1**n) + (A**n))) * F(L,S,C) - kd*B
    dXdt = alpha*(K2**h/(((((A - B)/(A + B)) + 1)/2)**h + K2**h)) - X
    
    #Returning Each
    return(dLdt,dAdt,dBdt,dXdt)

##############################################################################

'Single Run'

# This function takes in the initial conditions for the ODE and integrates.
# It then returns whether the cell has lived [0], died from osmotic crisis [1],
# or ended up running out of its metabolites [-1]. In assessing the cause of 
# death, I took advantage of what I knew about the phase portrait since decay
# corresponded to a asymptotic state and bursting was in the transient. 

# Note that the initial condition for 'L' is set within this function. I have
# values commented for 0, 10, and -10, but anything could be placed here. 

def survival(a,b):
    
    
    # Setting the death variable to some absurd value so it will not be saved
    dead = 1000.0
    
    # Setting the initial 'L' and 'X' conditions
    #init_L = 0.0
    #init_L = 10.0
    init_L = -10.0
    
    init_X = 0.5
    
    # init_A = np.linspace(0,20,50)
    # init_B = np.linspace(0,20,50)
    
    # for pos_a, a in enumerate(init_A):
    #     for pos_b, b in enumerate(init_B):
            
    x0 = [init_L,a,b,init_X] # Initial conditions

    #t = np.linspace(0,200,1000) # Duration and resolution of the simulation
    t = np.linspace(0,800,2000)

    x = odeint(ode_func, x0, t)
    
    L = x[:,0]
    A = x[:,1]
    B = x[:,2]
    X = x[:,3]
    
    C = A+B
    
    if (all(a > 0.1 for a in A) and all(b > 0.1 for b in B) and all(c <= 20 for c in C)): 
        dead = 0.0
    else:
        if not all(c <= 20 for c in C):
            dead = 1.0
        else:
            if not (all(a > 0.1 for a in A) and all(b > 0.1 for b in B)):
                dead = -1.0
                    
    return(dead)

##############################################################################

'Helper Function'

# This function is used to make it so that the parallel processing can be done.
# It is a necessary step dictated by the way the tool works.

def helper(z):
    A, B, posit_A, posit_B = z
    status = survival(A, B)
    return (status, posit_A, posit_B)

##############################################################################

# This function makes it so that all the various inputs for the single run 
# become one tuple so that we can use parallel processing with it. 

def adapt_matrix(resolution):
    
    # Initializing A's and B's that need to be scanned over.
    init_A = np.linspace(0,20,resolution)
    init_B = np.linspace(0,20,resolution)
    init_B = np.flip(init_B)
    conds = []
    
    # The matrix that the status of death or survival needs to be saved into.
    adpt_matrix = np.zeros([resolution,resolution])    
    
    # Making a series of tuple inputs for the helper function. 
    for pos_a, a in enumerate(init_A):
        for pos_b, b in enumerate(init_B):
            conds.append((a,b,pos_a,pos_b))
            
    # Attempting parallel processing.
    if __name__ == '__main__':
        num_workers = multip.cpu_count()
        pool = multip.Pool(num_workers)
        #pool = multip.Pool(processes=4)
        outputs = pool.map(helper, conds)
        #print("Input: {}".format(inputs))
        #print("Output: {}".format(outputs))
            
        for i in outputs: # Scanning through the outputs...
            position_a = i[1] # Look at the matrix position relative to 'A'
            position_b = i[2] # Look at the matrix position relative to 'B'
            adpt_matrix[position_b,position_a] = i[0] # Assigning the survival/death
             
    
        return(adpt_matrix)

    
##############################################################################

'Mapping Viability at Resolution 100'

# Creating dataset
viability = adapt_matrix(1000)

#############################################################################

'Visualizing the Data'

## If you would like to visualize the data, you can run this code. Note that
## this is assuming that the resolution is 1000. For whatever reason, the 
## the computer sometimes gets unhappy when things are all ran together, which
## is why I have this commented out. You can also just save the dataset and 
## visualize it in a different program.

# im = plt.imshow(viability, cmap='binary')
# plt.title('Protocell Survival (Initial L=10.0)')
# plt.xlabel('[A]')
# plt.ylabel('[B]')
# plt.axhline(990, color='red')
# plt.axvline(10, color='red')
# plt.plot([0,999],[0,999])
# plt.xticks([0,250,500,750,1000], [0,5,10,15,20])
# plt.yticks([0,250,500,750,1000], [20,15,10,5,0])
# plt.rcParams["figure.dpi"] = 800
# plt.show()