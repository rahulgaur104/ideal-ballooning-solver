#!/usr/bin/env python3
"""
In this file the user sets all the important paramters used for stability optimization
"""

import numpy as np
import os
import pickle
import pdb
import sys

eqbm_option =int(eval(sys.argv[1]))

#max function evaluation before termination
maxf = int(35)

#Tip: You should try to choose boundary indices such that (totalndofs+1) is 
# a composite number

## Pick the equilibrium you want to stabilize
# eqbm_option = 0 => D3D
# eqbm_option = 1 => NCSX
# eqbm_option = 2 => HBERG
# eqbm_option = 3 => NW

if eqbm_option == int(0):
    # For the axisymmetric case, these aren't really pol and tor indices
    pol_idxs = np.array([1, 2, 3])
    tor_idxs = np.array([1, 3, 5]) 
elif eqbm_option == int(1) or eqbm_option == int(2): # NCSX or HBERG
    pol_idxs = np.array([0, 1, 2, 3, 4, 5, 6])
    # Range of toroidal indices for each poloidal index, for m = pol_idxs[i], n = [-tor_idx[i], tor_idx[i]]
    tor_idxs = np.array([4, 3, 3, 2, 2, 2, 1]) 
elif eqbm_option == int(3): # NW high-beta
    pol_idxs = np.array([0, 1, 2])
    tor_idxs = np.array([2, 1, 1]) 
else: # eqbm_option cold start
    pol_idxs = np.array([0, 1, 2])
    tor_idxs = np.array([2, 1, 1]) 

#indices of iota Dofs
iotaidxs = np.array([])
#iotaidxs = np.array([0, 1])
#iotaidxs = np.array([0, 3, 5])

# number of boundary degrees of freedom
ndofsb = 0
if eqbm_option == 0:
    ndofsb += len(pol_idxs) + len(tor_idxs)
else:
    for i in range(len(pol_idxs)):
        if pol_idxs[i] == 0:
            ndofsb += 2*int(tor_idxs[0])
        else:
            ndofsb += 2*(2*tor_idxs[i] + 1)

#number of iota/current degrees of freedom
ndofsi = len(iotaidxs)

#number of pressure degrees of freedom
ndofsp = int(0)

#Is phiedge free?
isphifree = 0

## absolute step size 
abs_step = 1.E-3

## relative step size
rel_step = 2.0E-3

#total number of degrees of freedon
totalndofs = ndofsb + ndofsi + ndofsp + isphifree

# number of flux surfaces to scan
nsurfs = int(8)

# nuber of processors on each node on your system
nprocspernode = int(96)

##########################################################################################
####################---------------BALLOONING SOLVER KNOBS---------------#################
##########################################################################################

#total number of python executables 
totalnexecball   = int(totalndofs+1)

# total number of jobs submitted
if eqbm_option == 0:
    # total number of nodes used
    njobsball = int(1)
    nodesperball = int(1)
else:
    njobsball = int(2)
    nodesperball = int(8)

ngroups = int(nodesperball/njobsball*nprocspernode/totalnexecball)
nsurfs  = int(ngroups) 

# Tells us how many alpha values can be scanned simultaneously in a single
# instance of ball_scan.py. np.floor is the same as np.int
nalpha_fac = np.floor(np.floor((nprocspernode*nodesperball)/totalnexecball)*1/nsurfs)

if nalpha_fac <= 2:
    print(nalpha_fac, "You may want to increase nodesperball for a faster ball scan!")

# number of processors per ball_scan  run
if np.mod((nprocspernode*nodesperball), totalnexecball*nsurfs) == 0:
    nprocsperball = int((nprocspernode*nodesperball)/totalnexecball)
elif int((nprocspernode*nodesperball)/(totalnexecball*nsurfs)) == 0:
    print("\nBall error: Error! Increase nodesperball! \n")
else:
    print("\nBall warning: not perfectly divisible! \n")


##########################################################################################
###################-------------ADDITIONAL UNRELATED KNOBS-------------###################
##########################################################################################

# username on the machine 
username  =  "rg6256"

## number of nodes needed for a simsopt scan (mod(nprocspernode*nodespersimsopt,12) = 0)
nodespersimsopt = int(1) 

ngroups        = int(nodespersimsopt*nprocspernode/((totalndofs + 1)*nsurfs))
# number of processors per ball_scan  run
if np.mod(nodespersimsopt*nprocspernode, (totalndofs + 1)*nsurfs) == 0:
    ngroups    = int(nodespersimsopt*nprocspernode/((totalndofs + 1)*nsurfs))
elif int(ngroups) == 0:
    print("\nSimsopt error: Error! Increase nodespersimsopt! \n")
else:
    print("\nSimsopt warning: not perfectly divisible! \n")


save_dict = {'maxf':maxf, 'eqbm_option':eqbm_option, 'pol_idxs':pol_idxs, 'tor_idxs':tor_idxs, 'iotaidxs':iotaidxs, 'isphifree':isphifree, 'totalndofs':totalndofs, 'nsurfs':nsurfs, 'abs_step':abs_step, 'rel_step':rel_step, 'username':username,  'nprocspernode':nprocspernode, 'nodesperball':nodesperball, 'totalnexecball':totalnexecball, 'njobsball':njobsball, 'nodespersimsopt':nodespersimsopt}

print(save_dict)

with open('params_dict.pkl', 'wb') as f:
    pickle.dump(save_dict, f)





