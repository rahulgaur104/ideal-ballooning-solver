#!/usr/bin/env python
"""
This scripts calculate the ballooning growth rates for all surfaces and for all Dofs.

"""
import numpy as np
import os
from mpi4py import MPI
from simsopt.util.mpi import MpiPartition, log
from simsopt.mhd.vmec import Vmec
from simsopt.mhd.profiles import Profile, ProfilePolynomial, ProfileScaled
from scipy.optimize import minimize
from simsopt.util.constants import ELEMENTARY_CHARGE
import subprocess as spr
import pickle
import sys
import time
import pdb
from utils import *


iter0 = int(eval(sys.argv[1]))
dof_idx = int(eval(sys.argv[2]))
ngroups = int(eval(sys.argv[3]))  # MPI groups in Simsopt

path0 = os.getcwd() + "/GS2_files"
path1 = os.getcwd() + "/save_n_load"

mpi = MpiPartition()
vmec = Vmec("input.dof{0}".format(dof_idx), keep_all_files=False)

with open("params_dict.pkl", "rb") as f:
    save_dict = pickle.load(f)

########################################################################################################
########################------------------------ SET x0-------------------------########################
########################################################################################################

# First we fix all the DOFs
vmec._boundary.fix_all()

if len(save_dict["pol_idxs"]) > 0:
    pol_idxs = save_dict["pol_idxs"]

if len(save_dict["tor_idxs"]) > 0:
    tor_idxs = save_dict["tor_idxs"]

vmec._boundary.fix_all()

if int(save_dict["eqbm_option"]) == 0:
    mpol = vmec.indata.mpol
    # Then we unfix our favorites
    # Unfix the even input rmnc boundary coefficients
    vmec._boundary.unfix(np.arange(1, 4, 1))
    # Unfix the even input zmns boundary coefficients
    vmec._boundary.unfix(np.arange(int(mpol + 1), int(mpol + 7), 2))

elif int(save_dict["eqbm_option"]) == 1 or int(save_dict["eqbm_option"]) == 2:
    mpol = vmec.indata.mpol
    ntor = vmec.indata.ntor
    for i in range(len(pol_idxs)):
        if pol_idxs[i] == 0:
            vmec._boundary.fixed_range(
                mmin=pol_idxs[i],
                mmax=pol_idxs[i],
                nmin=-0,
                nmax=tor_idxs[i],
                fixed=False,
            )
        else:
            vmec._boundary.fixed_range(
                mmin=pol_idxs[i],
                mmax=pol_idxs[i],
                nmin=-tor_idxs[i],
                nmax=tor_idxs[i],
                fixed=False,
            )
    vmec._boundary.fix("rc(0,0)")  # Major radius
else:
    mpol = vmec.indata.mpol
    ntor = vmec.indata.ntor
    idx_b = 2 * (mpol + 1) * ntor + 1
    # Unfixing ZBS coefficients
    for i in range(len(pol_idxs)):
        if i == 0 and tor_idxs[i] != 0:
            vmec._boundary.unfix(np.arange(idx_b, idx_b + tor_idxs[i], 1))
            idx_b = (
                idx_b + tor_idxs[i] + (ntor - tor_idxs[i]) + (ntor - tor_idxs[i + 1])
            )
        elif i == len(pol_idxs) - 1:
            vmec._boundary.unfix(np.arange(idx_b, idx_b + 2 * tor_idxs[i] + 1, 1))
        else:
            vmec._boundary.unfix(np.arange(idx_b, idx_b + 2 * tor_idxs[i] + 1, 1))
            idx_b = (
                idx_b
                + 2 * tor_idxs[i]
                + 1
                + (ntor - tor_idxs[i])
                + (ntor - tor_idxs[i + 1])
            )

    vmec._boundary.fix("rc(0,0)")  # Major radius


if len(save_dict["iotaidxs"]) > 0:

    vmec.indata.piota_type = "power_series"
    ai = np.abs(vmec.indata.ai)
    finit_coef_idx = len(ai[ai > 0])
    iotap = ProfilePolynomial(vmec.indata.ai[:finit_coef_idx] * 2.0)
    vmec.iota_profile = iotap
    iotap.unfix(save_dict["iotaidxs"])
    vmec.iota_profile = ProfileScaled(iotap, 0.5)


if save_dict["isphifree"] == 1:
    vmec.unfix("phiedge")
# vmec._boundary.unfix(np.arange(int(vmec.indata.mpol + 3), int(vmec.indata.mpol + 4) , 1))

abs_step = save_dict["abs_step"]
rel_step = save_dict["rel_step"]

isabs = 0

if iter0 == 0:
    if dof_idx == 0:  # For the original x0; No Dof is being changed
        # dof_idx = 0	 # redundant
        vmec.x = np.load(path1 + "/x0.npy", allow_pickle=True)
    elif np.abs(vmec.x[dof_idx - 1]) <= 1e-2:  # If a Dof < 0.01, take an absolute step
        # y = vmec.x
        y = np.load(path1 + "/x0.npy", allow_pickle=True)
        y[dof_idx - 1] = y[dof_idx - 1] + abs_step
        vmec.x = y
        # vmec.x[dof_idx-1] = vmec.x[dof_idx-1] + abs_step
        isabs = 1
    else:  # otherwise take a relative step
        # y = vmec.x
        y = np.load(path1 + "/x0.npy", allow_pickle=True)
        y[dof_idx - 1] = y[dof_idx - 1] * (1 + rel_step)
        vmec.x = y
        # vmec.x[dof_idx-1] = vmec.x[dof_idx-1]*(1 + rel_step)
else:
    if dof_idx == 0:  # For the original x0; No Dof is being changed
        vmec.x = np.load(path1 + "/x0.npy", allow_pickle=True)[-1]
    elif (
        np.abs(np.load(path1 + "/x0.npy", allow_pickle=True)[-1][dof_idx - 1]) <= 1e-2
    ):  # If a Dof is zero, take an absolute step
        y = np.load(path1 + "/x0.npy", allow_pickle=True)[-1]
        y[dof_idx - 1] = y[dof_idx - 1] + abs_step
        vmec.x = y
        # vmec.x[dof_idx-1] = vmec.x[dof_idx-1] + abs_step
        isabs = 1
    else:  # otherwise take a relative step
        y = np.load(path1 + "/x0.npy", allow_pickle=True)[-1]
        y[dof_idx - 1] = y[dof_idx - 1] * (1 + rel_step)
        vmec.x = y
        # vmec.x[dof_idx-1] = vmec.x[dof_idx-1]*(1 + rel_step)


# pdb.set_trace()
##############################################################################################################
############-----------SET MPI GROUPS, RUN VMEC, CALCULATE COEFICIENTS AND SAVE GRID FILE---------############
##############################################################################################################
## Mpi communicator
comm = MPI.COMM_WORLD
# if comm.Get_rank() == 0:
#    print("DOFs = %d"%(len(vmec.x)))
#    print("size = %d"%(comm.Get_size()))

print("dof_chk before MPI", dof_idx)
# ngroups is equal to the number of surfaces
mpi = MpiPartition(ngroups)
vmec.verbose = mpi.proc0_world
# print(flush=True)
vmec.mpi = mpi
vmec.fcomm = mpi.comm_groups.py2f()
vmec.iter = mpi.comm_world.bcast(vmec.iter)

comm_grp = mpi.comm_groups
rank = comm_grp.Get_rank()
size = comm_grp.Get_size()

# pdb.set_trace()
comm_lead = mpi.comm_leaders
# print(rank, mpi.group)
# vmec       = Vmec("wout_D3D_negtri_pres_scale_10_hres_{0}.nc".format(dof_idx))

print("dof_chk", dof_idx)

vs = vmec_splines(vmec)
# The number of surfaces nsurfs = ngroups
# For each group there are multiple processors. For 3D cases, we will use these processes
# to parallelize the alpha scan. For 2D, we don't exploit these processes
nsurfs = ngroups

# rho_arr = np.linspace(0.4, 0.95, nsurfs)
rho_arr = np.linspace(0.5, 0.95, nsurfs)

# nperiod    = 3
# theta_fac  = 2*nperiod-1
theta_fac = int(4)

if vmec.wout.ntor == 0:
    ntheta = int(2 * vmec.wout.mpol * theta_fac) + 1
else:
    ntheta = int(2 * vmec.wout.mpol * vmec.wout.ntor * theta_fac) + 1

theta = np.linspace(-theta_fac * np.pi, theta_fac * np.pi, ntheta)
vguess = (1 - np.tanh(theta[1:-1] / np.pi) ** 2) * np.cos(theta[1:-1] / (2 * theta_fac))


temp_arr1 = np.zeros((nsurfs,))
theta0_arr = np.ones((nsurfs,)) * 0.1
alpha_arr = np.ones((nsurfs,)) * 0.1

# each group has mulitple processors. These arrays store data on from different field line scans.
temp_arr1_s = np.zeros((nsurfs,))
temp_arr2_s = np.zeros((nsurfs, 5, ntheta))
theta0_arr_s = np.ones((nsurfs,)) * 0.1
alpha_arr_s = np.ones((nsurfs,)) * 0.1
# pdb.set_trace()

ntheta0_guess = 15
nalpha_guess = 24
theta0_scan_guess = np.linspace(0.0, 0.5 * np.pi, ntheta0_guess)
alpha_scan_guess = np.linspace(0, np.pi, nalpha_guess)

# if rank == 0: # rank 0 of each group. For 3D case each rank would correspond to a different alpha
temp_RE = 0.0
temp_RE_guess = 1.0

# Iterative eigenvalue solvers need good guesses for fast convergence
vguess = (1 - np.tanh(theta[1:-1] / np.pi) ** 2) * np.cos(theta[1:-1] / (2 * theta_fac))

alpha_grp = int(nalpha_guess / size)

print("size", size, nalpha_guess, dof_idx)

gamma_scan_guess = np.zeros((nalpha_guess, ntheta0_guess))
vguess_scan_guess = np.zeros((nalpha_guess, ntheta0_guess, ntheta - 2))
idx_max_gam = np.zeros((1,))


alpha_guess = np.array([0.0])
theta0_guess = np.array([0.0])
sigma0 = np.array([0.05])

for i in range(alpha_grp):
    # print("alpha_grp", i)
    aidx = int(i * size + rank)
    geo_coeffs = vmec_fieldlines(
        vs, rho_arr[mpi.group], alpha_scan_guess[aidx], theta1d=theta
    )
    bmag = geo_coeffs.bmag[0][0]
    gbdrift = geo_coeffs.gbdrift[0][0]
    cvdrift = geo_coeffs.cvdrift[0][0]
    cvdrift0 = geo_coeffs.cvdrift0[0][0]
    gds2 = geo_coeffs.gds2[0][0]
    gds21 = geo_coeffs.gds21[0][0]
    gds22 = geo_coeffs.gds22[0][0]
    gradpar = geo_coeffs.gradpar_theta_pest[0][0]
    dPdrho = -1.0 * 0.5 * np.mean((cvdrift - gbdrift) * bmag**2)

    # Find the max gamma on the coarse theta0_grid for a coarse alpha grid
    for j in range(ntheta0_guess):
        theta0_guess = theta0_scan_guess[j]
        cvdrift_fth = cvdrift + theta0_guess * cvdrift0
        gds2_fth = gds2 + 2 * theta0_guess * gds21 + theta0_guess**2 * gds22
        temp_RE, X_arr, _, _, _, _ = gamma_ball_full(
            dPdrho, theta, bmag, gradpar, cvdrift_fth, gds2_fth, vguess, temp_RE_guess
        )
        vguess = X_arr[1:-1]
        vguess_scan_guess[aidx, j] = vguess
        gamma_scan_guess[aidx, j] = temp_RE
        # temp_RE_guess            = 1.2*abs(temp_RE) + 0.05

# When the maximum growth rate is 0., we observe multiple indices with a growth rate of 0.
# This breaks the code. Therefore, we add the following if statement.
if np.max(gamma_scan_guess) == 0.0:
    alpha_guess = np.array([0.0])
    theta0_guess = np.array([0.0])
    sigma0 = np.array([0.05])
elif len(np.where(gamma_scan_guess == np.max(gamma_scan_guess))[0]) > 1:
    idx_max_gam0 = np.where(gamma_scan_guess == np.max(gamma_scan_guess))
    idx_max_gam = (np.array([idx_max_gam0[0][0]]), np.array([idx_max_gam0[1][0]]))
    alpha_guess = alpha_scan_guess[idx_max_gam[0]]
    theta0_guess = theta0_scan_guess[idx_max_gam[1]]
    vguess = vguess_scan_guess[idx_max_gam]
    sigma0 = 1.3 * np.abs(gamma_scan_guess[idx_max_gam]) + 0.05
else:
    idx_max_gam = np.where(gamma_scan_guess == np.max(gamma_scan_guess))
    alpha_guess = alpha_scan_guess[idx_max_gam[0]]
    theta0_guess = theta0_scan_guess[idx_max_gam[1]]
    vguess = vguess_scan_guess[idx_max_gam]
    sigma0 = 1.3 * np.abs(gamma_scan_guess[idx_max_gam]) + 0.05


# if rank >= 10:
#    print(k, rank, sigma0, gamma_scan_guess)

print("info before rank = ", rank, sigma0, dof_idx)

## After coarsely scanning the growth rate over alpha-theta0 grid, we use the maximum growth rate from
## the coarse grid and use that to find the accurate maximum growth rate and the correspondgin alpha and theta0
if rank == 0:
    # Now, we launch a local optimizer from the coarse global maximum
    obj1 = minimize(
        obj_w_grad,
        x0=(alpha_guess[0], theta0_guess[0]),
        args=(vs, rho_arr[mpi.group], theta, vguess, sigma0),
        jac=True,
        bounds=((0.0, np.pi), (0.0, 0.5 * np.pi)),
        options={"ftol": 5.0e-11, "gtol": 2.0e-08, "maxiter": 30},
    )

    # values of alpha and theta0 at which the growth rate is maximum
    alpha_guess = obj1.x[0]
    theta0_guess = obj1.x[1]

    print("max_gamma, mpi.group, dof_idx", mpi.group, dof_idx, obj1)

    geo_coeffs = vmec_fieldlines(vs, rho_arr[mpi.group], alpha_guess, theta1d=theta)
    aidx = 0
    bmag = geo_coeffs.bmag[0][aidx]
    gbdrift = geo_coeffs.gbdrift[0][aidx]
    cvdrift = geo_coeffs.cvdrift[0][aidx]
    cvdrift0 = geo_coeffs.cvdrift0[0][aidx]
    gds2 = geo_coeffs.gds2[0][aidx]
    gds21 = geo_coeffs.gds21[0][aidx]
    gds22 = geo_coeffs.gds22[0][aidx]
    gradpar = geo_coeffs.gradpar_theta_pest[0][aidx]
    dPdrho = -1.0 * 0.5 * np.mean((cvdrift - gbdrift) * bmag**2)

    cvdrift_fth = cvdrift + theta0_guess * cvdrift0
    gds2_fth = gds2 + 2 * theta0_guess * gds21 + theta0_guess**2 * gds22

    temp_RE, X_arr, dX_arr, g_arr, c_arr, f_arr = gamma_ball_full(
        dPdrho, theta, bmag, gradpar, cvdrift_fth, gds2_fth, vguess
    )

    comm_lead.Barrier()

    # print("after barrier!")
    # gather theta0_arr_s from all the groups into a single array
    comm_lead.Gather([theta0_guess, MPI.DOUBLE], [theta0_arr_s, MPI.DOUBLE], root=0)
    comm_lead.Gather([alpha_guess, MPI.DOUBLE], [alpha_arr_s, MPI.DOUBLE], root=0)
    comm_lead.Gather([temp_RE, MPI.DOUBLE], [temp_arr1_s, MPI.DOUBLE], root=0)

    if (
        mpi.group == 0
    ):  # Collecting all the growth rates from different surfaces/mpi.groups and
        # saving them to a numpy file

        # temp_arr2  =  temp_arr2_s
        temp_arr1 = temp_arr1_s
        theta0_arr = theta0_arr_s
        alpha_arr = alpha_arr_s

        Ltc0 = np.load(
            path1 + "/ball_gam{0}.npy".format(int(dof_idx)), allow_pickle=True
        )
        theta0tc0 = np.load(
            path1 + "/ball_theta0{0}.npy".format(int(dof_idx)), allow_pickle=True
        )
        alphatc0 = np.load(
            path1 + "/ball_alpha{0}.npy".format(int(dof_idx)), allow_pickle=True
        )

        if iter0 == 0:
            Ltc0 = np.append(Ltc0, temp_arr1)
            Ltc0 = np.delete(Ltc0, 0)
            theta0tc0 = np.append(theta0tc0, theta0_arr)
            theta0tc0 = np.delete(theta0tc0, 0)
            alphatc0 = np.append(alphatc0, alpha_arr)
            alphatc0 = np.delete(alphatc0, 0)
        else:
            Ltc0 = np.vstack((Ltc0, temp_arr1))
            theta0tc0 = np.vstack((theta0tc0, theta0_arr))
            alphatc0 = np.vstack((alphatc0, alpha_arr))

        # Save the theta0_arr and gamma_arr(temp_arr1) for each dof
        np.save(path1 + "/ball_gam{0}.npy".format(int(dof_idx)), Ltc0)
        np.save(path1 + "/ball_theta0{0}.npy".format(int(dof_idx)), theta0tc0)
        np.save(path1 + "/ball_alpha{0}.npy".format(int(dof_idx)), alphatc0)

        spr.call(["rm -r dcon*.txt\n rm -r threed1.*"], shell=True)
