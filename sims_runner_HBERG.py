#!/usr/bin/env python
"""
This script performs derivative based optimization of the Henneberg-QA stellarator equilibrium
 against the ideal-ballooning mode using the SIMSOPT framework.

"""
import numpy as np
import subprocess as spr
from scipy.optimize import least_squares
import pickle
import pdb
import os

iter0 = int(0)

path0  = os.getcwd() + "/save_n_load"

# remove all the old files
spr.call(['python3 -u create_dict.py 2'], shell=True)

# create dictionary with all the sim-related information
spr.call(['python3 -u arr_create2.py'], shell=True)

with open("params_dict.pkl", 'rb') as f:
    save_dict = pickle.load(f)

totalndofs  = save_dict['totalndofs']
nsurfs      = save_dict['nsurfs']

# create redistribution arrays
spr.call(['python3 -u arr_reset.py {0}'.format('f')], shell=True)

# set_x0
spr.call(['python3 -u set_x0_submit.py'], shell=True)

##load target values of the penalty terms
aminor0  = 0.6015
volavgB0 = 2.500
aspect0  = 3.3739
ithresh0 = 0.63

with open(path0 + "/penalty.npy", 'wb') as f:
    np.save(f, np.array([aminor0, volavgB0, aspect0, ithresh0]))

#prefactor array. 
#0 -> minor,
#1 -> <B>,
#2 -> aspect,
#3 -> iotath,
#4 -> R_c,
#5 -> f_qs,
#6 -> micro_gamma,
#7 -> ball_gamma

gamma_ball_thresh = -0.0003
prefac = np.array([0.1, 0.2, 0.0, 0., 0., 0., 0., 50])


with open(path0 + "/penalty_prefac.npy", 'wb') as f:
    np.save(f, prefac)

# get x0/set x0
x0  = np.load(path0 + "/x0.npy", allow_pickle=True)

df0 = np.zeros((totalndofs, ))

with open('params_dict.pkl', 'rb') as f:
    save_dict = pickle.load(f)


def dfobj(x0):
    global iter0
    
    f0_arr   = np.zeros((totalndofs+1,))
    df0_arr  = np.zeros((1, totalndofs+1))
    step_arr = np.zeros((totalndofs+1,))
    
    if len(np.shape(np.load(path0 + "/x0.npy", allow_pickle=True))) == 1:
        x0_old = np.load(path0 + "/x0.npy", allow_pickle=True)
    else:
        x0_old = np.load(path0 + "/x0.npy", allow_pickle=True)[-1]
    
    #print("x0_old and x0", x0_old, x0, iter0)

    gamma_gthrd = np.zeros((nsurfs,))
    ky_max_gthrd = np.zeros((nsurfs,))
    kx_max_gthrd = np.zeros((nsurfs,))

    gamma_gthrd2 = np.zeros((totalndofs+1, nsurfs))
    gamma_ball2  = np.zeros((totalndofs+1, nsurfs))


    # if the optimizer asks for the gradient at a different value of x0
    # we recalculate the new eqbm and new objective function
    if np.array_equal(x0, x0_old) == False:
        
        # stack x0 only when x0 is changed
        x0_old = np.vstack((x0_old, x0))
        np.save(path0 + "/x0.npy", x0_old)

        spr.call(['python3 -u arr_reset.py {0}'.format('f')], shell=True)

        spr.call(['python3 -u Simsopt_submit.py {0}'.format(iter0)], shell=True)

        isconvrgd = np.load(path0 + "/isconvrgd.npy", allow_pickle=True).item()
        
        if isconvrgd == 1:

            spr.call(['python3 -u ball_submit.py {0}'.format(iter0)], shell=True)

            for i in range(totalndofs+1):
                dof_idx = i
                
                if np.load(path0 + '/isabs{0}.npy'.format(i))[-1] == 1: #never called at the first itern
            	    step_arr[i] = save_dict['abs_step']
                else:
            	    step_arr[i] = save_dict['rel_step']*x0[i-1]
            
                gamma_ball     = np.load(path0 + "/ball_gam{0}.npy".format(dof_idx))[-1]
                gamma_ball2[i] = gamma_ball
            #gamma_ball = np.array([0.])


        else:# What should the gradients be if VMEC doesn't converge? Setting to 0
            for i in range(totalndofs+1):

                dof_idx = i

                if np.load(path0 + '/isabs{0}.npy'.format(i))[-1] == 1: #never called at the first itern
                	step_arr[i] = save_dict['abs_step']
                else:
                	step_arr[i] = save_dict['rel_step']*x0[i-1]
                
                gamma_ball     = np.load(path0 + "/ball_gam{0}.npy".format(dof_idx))[-1]
                gamma_ball2[i] = gamma_ball*0
                #gamma_ball = np.array([0.])
   
    else: # x0 is the same as x0 for fobj

        isconvrgd = np.load(path0 + "/isconvrgd.npy", allow_pickle=True).item()

        if isconvrgd == 1:
            spr.call(['python3 -u arr_reset.py {0}'.format('p')], shell=True)
            
            for i in range(totalndofs+1):
                dof_idx = i

                if np.load(path0 + '/isabs{0}.npy'.format(i), allow_pickle=True)[-1] == 1:
                	step_arr[i] = save_dict['abs_step']
                else:
                	step_arr[i] = save_dict['rel_step']*x0[i-1]

                gamma_ball = np.load(path0 + "/ball_gam{0}.npy".format(dof_idx), allow_pickle=True)
                if len(np.shape(gamma_ball)) == 1:
                    gamma_ball2[i] = gamma_ball
                else:
                    gamma_ball2[i] = gamma_ball[-1]
        else:# What should the gradients be if VMEC doesn't converge? Setting to 0
            for i in range(totalndofs+1):
                if len(np.shape(gamma_ball)) == 1:
                    gamma_ball2[i] = np.zeros((nsurfs,))
                else:
                    gamma_ball2[i] = np.zeros((nsurfs,))


    if isconvrgd == 1:
        for i in range(totalndofs+1):
            # load the incomplete objective function 
            f0  = np.load(path0 + "/f{0}.npy".format(i), allow_pickle=True)[-1]
            # The overall objective function
            f0  = f0  + prefac[-1]*np.sum(np.maximum(gamma_ball2[i] - gamma_ball_thresh , 0.))
            f0_arr[i] = f0
            if i > 0:
                df0_arr[0, i] = (f0_arr[i] - f0_arr[0])/step_arr[i] * 0.5 * 1/np.sqrt(f0_arr[0])
    
    ## Use this if the objective function has a square root
    #df0_arr = (f0_arr[1:] - f0_arr[0])/step_arr[1:] * 0.5 * 1/np.sqrt(f0_arr[0])
    
    f0_list = f0_arr.tolist()
    f0 = open('f0_list.out', 'a')
    f0.write(f'{iter0}, {f0_list}')
    f0.write('\n')
    f0.close()

    df0_list = df0_arr[0, :].tolist()
    df0 = open('df0_list.out', 'a')
    df0.write(f'{iter0}, {df0_list}')
    df0.write('\n')
    df0.close()

    #print("f0_arr = \n", f0_arr)
    #print("df0_arr = \n", df0_arr)
    return df0_arr[:, 1:]

def fobj(x0):
    global iter0
    
    dof_idx = int(0)

    # saving x0 so that it can be read later
    if iter0 > 0:
    	P0 = np.load(path0 + "/x0.npy", allow_pickle=True)
    	P0 = np.vstack((P0, x0))
    	np.save(path0 + "/x0.npy", P0)
    
    spr.call(['python3 -u arr_reset.py {0}'.format('f')], shell=True)

    spr.call(['python3 -u Simsopt_submit.py {0}'.format(iter0)], shell=True)
       
    isconvrgd = np.load(path0 + "/isconvrgd.npy", allow_pickle=True).item()

    if isconvrgd == 1:
        #spr.call(['rm -r slurm-*.out'], shell=True)

        spr.call(['python3 -u ball_submit.py {0}'.format(iter0)], shell=True)
        if iter0 == 0:
            gamma_ball = np.load(path0 + "/ball_gam{0}.npy".format(dof_idx), allow_pickle=True)
        else:
            gamma_ball = np.load(path0 + "/ball_gam{0}.npy".format(dof_idx), allow_pickle=True)[-1]

        iter0 = iter0 + 1
        # load the incomplete objective function 
        f0  = np.load(path0 + "/f0.npy", allow_pickle=True)[-1]
        
        # The overall objective function
        f0  = f0 + prefac[-1]*np.sum(np.maximum(gamma_ball - gamma_ball_thresh, 0.))
    else:
        f0 = 9999.
 
    print("obj f0 = ", f0)
   
    return np.sqrt(f0)

	

iota_lb = np.array([-0.40, -0.5, -0.5]) # additional factor due to scaling
iota_ub = np.array([0.625, 0.2, 0.4])


mpol_max    = int(9)
ntor_max    = int(9)

RBClb       = np.zeros((9, 9))
RBCub       = np.zeros((9, 9))
ZBSlb       = np.zeros((9, 9))
ZBSub       = np.zeros((9, 9))

RBClb[0, :] = np.array([  1.9, -0.40, -0.15, -0.10, -0.05,  0.00,  0.00,  0.00,  0.00]) 
RBClb[1, :] = np.array([-0.05, -0.05, -0.05, -0.25,  0.05, -0.55, -0.05, -0.03, -0.03]) 
RBClb[2, :] = np.array([-0.05, -0.05, -0.05, -0.10, -0.05, -0.10, -0.05, -0.05, -0.05]) 
RBClb[3, :] = np.array([-0.05, -0.05, -0.05, -0.06, -0.15, -0.05, -0.12, -0.05, -0.05]) 
RBClb[4, :] = np.array([-0.04, -0.04, -0.04, -0.07, -0.07, -0.07, -0.07, -0.04, -0.04]) 
RBClb[5, :] = np.array([-0.03, -0.04, -0.05, -0.08, -0.08, -0.08, -0.08, -0.05, -0.03]) 
RBClb[6, :] = np.array([-0.01, -0.01, -0.02, -0.05, -0.06, -0.05, -0.05, -0.03, -0.01]) 

RBCub[0, :] = np.array([ 2.1,  0.40,  0.15,  0.10,  0.05,  0.00,  0.00,  0.00, 0.00]) 
RBCub[1, :] = np.array([0.05,  0.05,  0.10,  0.20,  0.80,  0.25,  0.10,  0.05, 0.03]) 
RBCub[2, :] = np.array([0.05,  0.05,  0.15,  0.20,  0.30,  0.40,  0.15,  0.10, 0.05])
RBCub[3, :] = np.array([0.05,  0.05,  0.08,  0.10,  0.10,  0.15,  0.15,  0.10, 0.05]) 
RBCub[4, :] = np.array([0.02,  0.04,  0.04,  0.07,  0.07,  0.07,  0.05,  0.02, 0.02])
RBCub[5, :] = np.array([0.03,  0.03,  0.05,  0.04,  0.10,  0.10,  0.08,  0.05, 0.03]) 
RBCub[6, :] = np.array([0.01,  0.01,  0.02,  0.02,  0.04,  0.02,  0.02,  0.01, 0.01]) 


ZBSlb[0, :] = np.array([ 0.00, -0.25, -0.15, -0.10, -0.05,  0.00,  0.00,  0.00,  0.00]) 
ZBSlb[1, :] = np.array([-0.02, -0.03, -0.10, -0.15, -0.15, -0.10, -0.05, -0.04, -0.03]) 
ZBSlb[2, :] = np.array([-0.02, -0.03, -0.10, -0.15, -0.15, -0.20, -0.10, -0.08, -0.03]) 
ZBSlb[3, :] = np.array([-0.02, -0.03, -0.04, -0.06, -0.10, -0.10, -0.05, -0.03, -0.03]) 
ZBSlb[4, :] = np.array([-0.08, -0.02, -0.05, -0.05, -0.05, -0.05, -0.05, -0.02, -0.05]) 
ZBSlb[5, :] = np.array([-0.03, -0.02, -0.03, -0.03, -0.05, -0.03, -0.03, -0.02, -0.03]) 
ZBSlb[6, :] = np.array([-0.01, -0.01, -0.02, -0.02, -0.04, -0.02, -0.02, -0.01, -0.01]) 

ZBSub[0, :] = np.array([0.00,  0.25,  0.15,  0.10,  0.05,  0.00,  0.00, 0.00, 0.00]) 
ZBSub[1, :] = np.array([0.01,  0.05,  0.10,  0.15,  1.30,  0.60,  0.07, 0.04, 0.03]) 
ZBSub[2, :] = np.array([0.08,  0.10,  0.10,  0.15,  0.30,  0.20,  0.15, 0.10, 0.05]) 
ZBSub[3, :] = np.array([0.08,  0.10,  0.15,  0.30,  0.35,  0.20,  0.15, 0.10, 0.05]) 
ZBSub[4, :] = np.array([0.02,  0.04,  0.05,  0.05,  0.05,  0.05,  0.05, 0.04, 0.03]) 
ZBSub[5, :] = np.array([0.03,  0.02,  0.03,  0.03,  0.05,  0.03,  0.03, 0.02, 0.03]) 
ZBSub[6, :] = np.array([0.01,  0.01,  0.02,  0.02,  0.04,  0.02,  0.02, 0.01, 0.01]) 

pol_idxs = save_dict['pol_idxs']
tor_idxs = save_dict['tor_idxs'] 

boundary_lb = np.empty([], dtype=int)
boundary_ub = np.empty([], dtype=int)

for i in range(len(pol_idxs)):
    if pol_idxs[i] == 0:
        boundary_lb = np.append(boundary_lb, RBClb[i][np.arange(1, tor_idxs[0]+1, 1)])
        boundary_ub = np.append(boundary_ub, RBCub[i][np.arange(1, tor_idxs[0]+1, 1)])
    else:                                            
        boundary_lb = np.append(boundary_lb, RBClb[i][int((mpol_max-1)/2) + np.arange(-tor_idxs[i], tor_idxs[i]+1, 1)])
        boundary_ub = np.append(boundary_ub, RBCub[i][int((mpol_max-1)/2) + np.arange(-tor_idxs[i], tor_idxs[i]+1, 1)])
    #print(len(boundary_lb))


for i in range(len(pol_idxs)):
    if pol_idxs[i] == 0:
        boundary_lb = np.append(boundary_lb, ZBSlb[i][np.arange(1, tor_idxs[0]+1, 1)])
        boundary_ub = np.append(boundary_ub, ZBSub[i][np.arange(1, tor_idxs[0]+1, 1)])
    else:
        boundary_lb = np.append(boundary_lb, ZBSlb[i][int((mpol_max-1)/2) + np.arange(-tor_idxs[i], tor_idxs[i]+1, 1)])
        boundary_ub = np.append(boundary_ub, ZBSub[i][int((mpol_max-1)/2) + np.arange(-tor_idxs[i], tor_idxs[i]+1, 1)])
    #print(len(boundary_lb))

#pdb.set_trace()

boundary_lb = np.delete(boundary_lb, 0)
boundary_ub = np.delete(boundary_ub, 0)

lb = boundary_lb
ub = boundary_ub

# wrap fobj in scipy.least_squares (local, gradient-based)
least_squares(fobj, x0, jac = dfobj, bounds = (lb, ub), diff_step = 1.0E-3, verbose=2, max_nfev = save_dict['maxf'], ftol = 5.E-5, xtol = 1.E-5)




