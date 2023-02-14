#!/usr/bin/env python

import numpy as np
import os
from scipy.integrate import simps
from scipy.sparse.linalg import eigs
import time
import pdb


#############################################################################################################
###################--------------------BALLOONING SOLVER FUNCTION--------------------------##################
#############################################################################################################


def gamma_ball_full(dPdrho, theta_PEST, B, gradpar, cvdrift, gds2, vguess):
    # Inputs  : geometric coefficients(normalized by a_N, and B_N)
    #           on an equispaced theta_PEST grid. The definition of 
    #           these coefficients can be obtained from Dr. Edmund Highcock's thesis
    #           
    # Outputs : The maximum ballooning growth rate gamma
    
    theta_ball = theta_PEST
    ntheta = len(theta_ball)
    
    #gds2 = (dpsidrho * |grad alpha|/(a_N*B_N))**2.
    #gradpar = a_N * (b dot grad theta)
    #cvdrift = a_N**2 * b cross (b dot grad b) dot grad alpha
    #rho = sqrt(s), s = normalized toroidal flux
    g = np.abs(gradpar)*gds2/(B)
    c = -1*dPdrho*cvdrift*1/(np.abs(gradpar)*B)
    f = gds2/B**2*1/(np.abs(gradpar)*B)
    
    len1 = len(g)
    
    ##Uniform half theta ball
    theta_ball_u = np.linspace(theta_ball[0], theta_ball[-1], len1)
    
    g_u = np.interp(theta_ball_u, theta_ball, g)
    c_u = np.interp(theta_ball_u, theta_ball, c)
    f_u = np.interp(theta_ball_u, theta_ball, f)
    
    # uniform theta_ball on half points with half the size, i.e., only from [0, (2*nperiod-1)*np.pi]
    # nperiod is a positive integer such that theta goes from (-(2*nperiod-1)*pi, (2*nperiod-1)*pi)
    theta_ball_u_half = (theta_ball_u[:-1] + theta_ball_u[1:])/2
    h = np.diff(theta_ball_u_half)[2]
    g_u_half = np.interp(theta_ball_u_half, theta_ball, g)
    g_u1 = g_u[:]
    c_u1 = c_u[:]
    f_u1 = f_u[:]
    
    len2 = int(len1)-2
    A  = np.zeros((len2, len2))
    
    # Asymmetric tridiagonal
    A = np.diag(g_u_half[1:-1]/f_u1[2:-1]*1/h**2, -1) +\
        np.diag(-(g_u_half[1:] + g_u_half[:-1])/f_u1[1:-1]*1/h**2 + c_u1[1:-1]/f_u1[1:-1], 0) +\
        np.diag(g_u_half[1:-1]/f_u1[1:-2]*1/h**2, 1)
    
    # Method without M is approx 3 X faster. Iterative Arnoldi solver.
    # Solution v can be used as a guess the next time solver is called (not demonstrated here)
    w, v  = eigs(A, 1, sigma=0.42, v0=vguess, tol=1E-6, OPpart='r')
    
    ### Variational refinement as described in Sanchez et al.
    X          = np.zeros((len2+2, ))
    dX         = np.zeros((len2+2, ))
    #X[1:-1]     = np.reshape(v[:, idx_max].real, (-1,))/np.max(np.abs(v[:, idx_max].real))
    X[1:-1]     = np.reshape(v[:, 0].real, (-1,))/np.max(np.abs(v[:, 0].real))
    
    X[0]       = 0.
    X[-1]      = 0.
    
    dX[0]      = (-1.5*X[0] + 2*X[1] - 0.5*X[2])/h
    dX[1]      = (X[2] - X[0])/(2*h)
    
    dX[-2]     = (X[-1] - X[-3])/(2*h)
    dX[-1]     = (0.5*X[-3] - 2*X[-2] + 1.5*0.)/(h)
    
    dX[2:-2]   = 2/(3*h)*(X[3:-1] - X[1:-3]) - (X[4:] - X[0:-4])/(12*h)
    
    Y0         = -g_u1*dX**2 + c_u1*X**2
    Y1         = f_u1*X**2
    
    # Simpson's 1/3 rule (different from Sanchez's original paper)
    return simps(Y0)/simps(Y1)


