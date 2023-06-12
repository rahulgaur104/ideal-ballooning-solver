#!/usr/bin/env python3
"""
The purpose of this script is to plot and/or save the s-alpha balooning stability diagradms for an equilibrium. This script needs the name of the dictionary containing all the information of a local equilibrium. 
Instead of alpha we choose dPdpsi

Note that this script only calculates the s-alpha diagram of a large aspect ratio tokamak. With minor modifications, we can also general s-alpha diagrams for an arbitrary tokamak.

The development of s-alpha calculator for a general stellarator is in progress.
"""

import os
import sys
import pdb
import numpy as np
import multiprocessing as mp
from scipy.interpolate import CubicSpline as cubspl
from matplotlib import pyplot as plt



def check_ball(shat_n, alpha_n, theta0):
        theta_ball = np.linspace(-61*np.pi, 61*np.pi, 1601)

        ntheta     = len(theta_ball)
        delthet = np.diff(theta_ball)

        diff = 0.0
        one_m_diff = 1 - diff

        # The expression of these coefficients are given in CHT's PRL, Bishop's notes, Edmund Highcock's thesis etc.
        g = 1 + (shat_n*(theta_ball-theta0) - alpha_n*(np.sin(theta_ball)-np.sin(theta0)))**2
        c = alpha_n*(np.cos(theta_ball) + np.sin(theta_ball)*(shat_n*(theta_ball-theta0) - alpha_n*(np.sin(theta_ball)-np.sin(theta0))))
        f = np.zeros((ntheta, ))

        ch = np.zeros((ntheta,))
        gh = np.zeros((ntheta,))
        fh = np.zeros((ntheta,))
        
        for i in np.arange(1, ntheta):
            ch[i] = 0.5*(c[i] + c[i-1]) 
            gh[i] = 0.5*(g[i] + g[i-1])
            fh[i] = 0.5*(f[i] + f[i-1])
        
        cflmax = np.max(np.abs(delthet**2*ch[1:]/gh[1:]))
        
        c1 = np.zeros((ntheta,))
        f1 = np.zeros((ntheta,))

        for ig in np.arange(1, ntheta-1):
            c1[ig] = -delthet[ig]*(one_m_diff*c[ig]+0.5*diff*ch[ig+1])\
                     -delthet[ig-1]*(one_m_diff*c[ig]+0.5*diff*ch[ig])\
                     -delthet[ig-1]*0.5*diff*ch[ig]
            c1[ig] = -delthet[ig]*(one_m_diff*c[ig])\
                     -delthet[ig-1]*(one_m_diff*c[ig])
            f1[ig] = -delthet[ig]*(one_m_diff*f[ig])\
                     -delthet[ig-1]*(one_m_diff*f[ig])
            c1[ig]=0.5*c1[ig]
            f1[ig]=0.5*f1[ig]
        
        
        c2 = np.zeros((ntheta,))
        f2 = np.zeros((ntheta,))
        g1 = np.zeros((ntheta,))
        g2 = np.zeros((ntheta,))
        
        for ig in np.arange(1, ntheta):
            c2[ig] = -0.25*diff*ch[ig]*delthet[ig-1]
            f2[ig] = -0.25*diff*fh[ig]*delthet[ig-1]
            g1[ig] = gh[ig]/delthet[ig-1]
            g2[ig] = 1.0/(0.25*diff*ch[ig]*delthet[ig-1]+gh[ig]/delthet[ig-1])
        
        
        psi_t = np.zeros((ntheta,))
        psi_t[1]=delthet[0]
        psi_prime=(psi_t[1]/g2[1])*0.5

 
        gamma = 0
        #for ig in np.arange(int((ntheta-1)/2),ntheta-1):
        for ig in np.arange(1,ntheta-1):
            #pdb.set_trace()
            psi_prime=psi_prime+1*c1[ig]*psi_t[ig]+c2[ig]*psi_t[ig-1]\
                      + gamma*(f1[ig]*psi_t[ig]+f2[ig]*psi_t[ig-1])
            psi_t[ig+1]=(g1[ig+1]*psi_t[ig]+psi_prime)*g2[ig+1]

        #pdb.set_trace()
        if np.isnan(np.sum(psi_t)) or np.isnan(np.abs(psi_prime)):
            print('warning NaN  balls')
        

        isunstable = 0
        for ig in np.arange(1,ntheta-1):
            if(psi_t[ig]*psi_t[ig+1] <= 0 ):
                isunstable = 1
                #print("instability detected... please choose a different equilibrium")
        return isunstable


def check_ball_long(shat_n, alpha_n, theta0):
        theta_ball = np.linspace(-20*np.pi, 20*np.pi, 401)
        ntheta     = len(theta_ball)
        delthet = np.diff(theta_ball)
        diff = 0.0
        one_m_diff = 1 - diff

        g = 1 + (shat_n*(theta_ball-theta0) - alpha_n*(np.sin(theta_ball)-np.sin(theta0)))**2
        c = alpha_n*(np.cos(theta_ball) + np.sin(theta_ball)*(shat_n*(theta_ball-theta0) - alpha_n*(np.sin(theta_ball)-np.sin(theta0))))
        f = np.zeros((ntheta, ))


        ch = np.zeros((ntheta,))
        gh = np.zeros((ntheta,))
        fh = np.zeros((ntheta,))
        
        for i in np.arange(1, ntheta):
            ch[i] = 0.5*(c[i] + c[i-1]) 
            gh[i] = 0.5*(g[i] + g[i-1])
            fh[i] = 0.5*(f[i] + f[i-1])
        
        cflmax = np.max(np.abs(delthet**2*ch[1:]/gh[1:]))
        
        c1 = np.zeros((ntheta,))
        f1 = np.zeros((ntheta,))

        for ig in np.arange(1, ntheta-1):
            c1[ig] = -delthet[ig]*(one_m_diff*c[ig]+0.5*diff*ch[ig+1])\
                     -delthet[ig-1]*(one_m_diff*c[ig]+0.5*diff*ch[ig])\
                     -delthet[ig-1]*0.5*diff*ch[ig]
            c1[ig] = -delthet[ig]*(one_m_diff*c[ig])\
                     -delthet[ig-1]*(one_m_diff*c[ig])
            f1[ig] = -delthet[ig]*(one_m_diff*f[ig])\
                     -delthet[ig-1]*(one_m_diff*f[ig])
            c1[ig]=0.5*c1[ig]
            f1[ig]=0.5*f1[ig]
        
        
        c2 = np.zeros((ntheta,))
        f2 = np.zeros((ntheta,))
        g1 = np.zeros((ntheta,))
        g2 = np.zeros((ntheta,))
        
        for ig in np.arange(1, ntheta):
            c2[ig] = -0.25*diff*ch[ig]*delthet[ig-1]
            f2[ig] = -0.25*diff*fh[ig]*delthet[ig-1]
            g1[ig] = gh[ig]/delthet[ig-1]
            g2[ig] = 1.0/(0.25*diff*ch[ig]*delthet[ig-1]+gh[ig]/delthet[ig-1])
        
        
        psi_t = np.zeros((ntheta,))
        psi_t[1]=delthet[0]
        psi_prime=(psi_t[1]/g2[1])*0.1
 
        gamma = 0
        for ig in np.arange(1,ntheta-1):
            #pdb.set_trace()
            psi_prime=psi_prime+1*c1[ig]*psi_t[ig]+c2[ig]*psi_t[ig-1]\
                      + gamma*(f1[ig]*psi_t[ig]+f2[ig]*psi_t[ig-1])
            psi_t[ig+1]=(g1[ig+1]*psi_t[ig]+psi_prime)*g2[ig+1]

        #pdb.set_trace()
        if np.isnan(np.sum(psi_t)) or np.isnan(np.abs(psi_prime)):
            print('warning NaN  balls')
        

        isunstable = 0
        for ig in np.arange(1,ntheta-1):
            if(psi_t[ig]*psi_t[ig+1] <= 0 ):
                isunstable = 1
                #print("instability detected... please choose a different equilibrium")
        return isunstable



# No of shat points
len1 = 200
# No of alpha_MHD(proportional to dpdpsi) points
len2 = 100

shat_grid = np.linspace(0., 2, len1)
alpha_grid = np.linspace(0., 1.2, len2)
ball_scan_arr1 = np.zeros((len1, len2))
ball_scan_arr2 = np.zeros((len1, len2))
ball_scan_arr3 = np.zeros((len1, len2))

os.environ["OMP_NUM_THREADS"]="1"

# Change the nop value to the number of processors on your system
nop = int(40)

pool = mp.Pool(processes=nop)
theta0_arr = np.array([0.0, 0.1, 0.2])
results1 = np.array([[pool.apply_async(check_ball, args=(shat_grid[i], alpha_grid[j], theta0_arr[0])) for i in range(len1)] for j in range(len2)])

for i in range(len2):
	ball_scan_arr1[:, i] = np.array([results1[i, j].get() for j in range(len1)])


results2 = np.array([[pool.apply_async(check_ball_long, args=(shat_grid[i], alpha_grid[j], theta0_arr[1])) for i in range(len1)] for j in range(len2)])

for i in range(len2):
	ball_scan_arr2[:, i] = np.array([results2[i, j].get() for j in range(len1)])


results3 = np.array([[pool.apply_async(check_ball_long, args=(shat_grid[i], alpha_grid[j], theta0_arr[2])) for i in range(len1)] for j in range(len2)])

for i in range(len2):
	ball_scan_arr3[:, i] = np.array([results3[i, j].get() for j in range(len1)])

from matplotlib import pyplot as plt

plt.figure(figsize=(5.71, 8.57))
Y, X = np.meshgrid(alpha_grid, shat_grid)

cs = plt.contour(Y, X, np.logical_or(np.logical_or(ball_scan_arr1, ball_scan_arr2)+0,ball_scan_arr3) , linewidths=2.5, colors='black', levels=[0.])  

plt.xticks(np.linspace(0, np.max(alpha_grid), 4), fontsize=16)
plt.yticks(np.linspace(0, np.max(shat_grid), 6), fontsize=16)

plt.xlabel(r"$\alpha$", fontsize=22)
plt.ylabel(r"$\hat{s}$", rotation=0, fontsize=22)

plt.text(0.8, 0.8 , "US", fontsize=22)
plt.text(0.2, 1.2 , "S", fontsize=22)
plt.text(1.0, 0.05 , "S", fontsize=22)

plt.tight_layout()
plt.savefig('test.png')
#plt.show()




