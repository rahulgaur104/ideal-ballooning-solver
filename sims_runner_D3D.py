#!/usr/bin/env python
"""
This script performs derivative based optimization of the D3D tokamak equilibrium
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
spr.call(['python3 -u create_dict.py 0'], shell=True)

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
aminor0  = 0.68
volavgB0 = 0.679
aspect0  = 2.42
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
#6 -> micro_gamma, # for microstability.
#7 -> ball_gamma

# set stability threshold. An equilibrium is ball stable if the 
# growth rate is less than gamma_ball_thresh
gamma_ball_thresh = -0.0002
prefac = np.array([1.0, 1.0, 1.0, 0., 1.0, 0., 0., 2])


with open(path0 + "/penalty_prefac.npy", 'wb') as f:
    np.save(f, prefac)

# get x0/set x0

x0  = np.load(path0 + "/x0.npy", allow_pickle=True)
#x0  = np.save(path0 + "/x0.npy", x0)


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

                gamma_ball = np.load(path0 + "/ball_gam{0}.npy".format(dof_idx), allow_pickle=True)[-1]
                gamma_ball2[i] = gamma_ball
        else:# What should the gradients be if VMEC doesn't converge? Setting to 0
            for i in range(totalndofs+1):

                gamma_ball2[i] = 0.


    if isconvrgd == 1:
        for i in range(totalndofs+1):
            # load the incomplete objective function 
            f0  = np.load(path0 + "/f{0}.npy".format(i), allow_pickle=True)[-1]
            # The overall objective function
            f0  = f0  + prefac[-1]*np.sum(np.maximum(gamma_ball2[i] - gamma_ball_thresh , 0.))
            f0_arr[i] = f0
            if i > 0:
                df0_arr[0, i] = (f0_arr[i] - f0_arr[0])/step_arr[i] * 0.5 * 1/np.sqrt(f0_arr[0])
        #df0_arr[0, i] = (f0_arr[i] - f0_arr[0])/step_arr[i]
    
    #pdb.set_trace()
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


RBClb       = np.zeros((9,))
RBCub       = np.zeros((9,))
ZBSlb       = np.zeros((9,))
ZBSub       = np.zeros((9,))

RBClb[:] = np.array([1.71, 0.4, -0.21, -0.12, -0.05, -0.00,  0.00,  0.00,  0.00]) 
RBCub[:] = np.array([1.73, 0.7,  0.31,  0.12,  0.05,  0.00,  0.00,  0.00,  0.00]) 

ZBSlb[:] = np.array([0.00, -0.92, -0.25, -0.20, -0.10, -0.10, -0.05,-0.05, -0.02]) 
ZBSub[:] = np.array([0.00, -0.20,  0.25,  0.20,  0.10,  0.10,  0.05, 0.05,  0.00]) 


# For the axisymmetric pol_idxs are the RBCs and tor_idxs are the ZBSs.
pol_idxs = save_dict['pol_idxs']
tor_idxs = save_dict['tor_idxs'] 

boundary_lb = np.concatenate((RBClb[pol_idxs], ZBSlb[tor_idxs]))
boundary_ub = np.concatenate((RBCub[pol_idxs], ZBSub[tor_idxs]))

#phiedge_lb = np.array([1.0])
#phiedge_ub = np.array([4.0])

#lb = np.concatenate((iota_lb, boundary_lb))
#ub = np.concatenate((iota_ub, boundary_ub))

lb = boundary_lb
ub = boundary_ub
	
# wrap fobj in scipy.least_squares (local, gradient-based)
least_squares(fobj, x0, jac = dfobj, bounds = (lb, ub), diff_step = 1.0E-3, verbose=2, max_nfev = save_dict['maxf'], ftol = 5.E-5, xtol = 1.E-5)
#least_squares(fobj, x0, jac = dfobj, diff_step = 5.0E-4, verbose=2, max_nfev = save_dict['maxf'], ftol = 5.E-5, xtol = 1.E-5)




