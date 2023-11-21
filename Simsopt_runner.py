#!/usr/bin/env python
"""
This script calls multiple instances of vmec, each with a different Dof and generates the GS2 grid file from the equilibrium. One typically has to call ndofs + 1 instances to calcuate the objective function and the jacobian for ndofs
"""
import numpy as np
import os
from mpi4py import MPI
from simsopt.util.mpi import MpiPartition, log
from simsopt.mhd.vmec import Vmec
from simsopt.mhd.vmec_diagnostics import QuasisymmetryRatioResidual
from simsopt.mhd.profiles import Profile, ProfilePolynomial, ProfileScaled
from simsopt.util.constants import ELEMENTARY_CHARGE
import subprocess as spr
import pickle
import sys
import time
import pdb

# from utils_axisym2 import *
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
######################------------------------ SET Dofs-------------------------########################
########################################################################################################


if len(save_dict["pol_idxs"]) > 0:
    pol_idxs = save_dict["pol_idxs"]

if len(save_dict["tor_idxs"]) > 0:
    tor_idxs = save_dict["tor_idxs"]

vmec._boundary.fix_all()

if int(save_dict["eqbm_option"]) == 0:
    mpol = vmec.indata.mpol
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


abs_step = save_dict["abs_step"]
rel_step = save_dict["rel_step"]

isabs = np.array([0], dtype=int)


if iter0 == 0:
    if dof_idx == 0:  # For the original x0; No Dof is being changed
        vmec.x = np.load(path1 + "/x0.npy", allow_pickle=True)
    elif (
        np.abs(np.load(path1 + "/x0.npy", allow_pickle=True)[dof_idx - 1]) <= 1e-2
    ):  # If a Dof < 0.01, take an absolute step
        y = np.load(path1 + "/x0.npy", allow_pickle=True)
        y[dof_idx - 1] = y[dof_idx - 1] + abs_step
        vmec.x = y
        isabs[0] = int(1)
    else:  # otherwise take a relative step
        y = np.load(path1 + "/x0.npy", allow_pickle=True)
        y[dof_idx - 1] = y[dof_idx - 1] * (1 + rel_step)
        vmec.x = y
else:
    if dof_idx == 0:  # For the original x0; No Dof is being changed
        vmec.x = np.load(path1 + "/x0.npy", allow_pickle=True)[-1]
    elif (
        np.abs(np.load(path1 + "/x0.npy", allow_pickle=True)[-1][dof_idx - 1]) <= 1e-2
    ):  # If a Dof is zero, take an absolute step
        y = np.load(path1 + "/x0.npy", allow_pickle=True)[-1]
        y[dof_idx - 1] = y[dof_idx - 1] + abs_step
        vmec.x = y
        isabs[0] = int(1)
    else:  # otherwise take a relative step
        y = np.load(path1 + "/x0.npy", allow_pickle=True)[-1]
        y[dof_idx - 1] = y[dof_idx - 1] * (1 + rel_step)
        vmec.x = y


# pdb.set_trace()
##############################################################################################################
############-----------SET MPI GROUPS, RUN VMEC, CALCULATE COEFICIENTS AND SAVE GRID FILE---------############
##############################################################################################################
## Mpi communicator
comm = MPI.COMM_WORLD
# if comm.Get_rank() == 0:
#    print("DOFs = %d"%(len(vmec.x)))
#    print("size = %d"%(comm.Get_size()))
mpi = MpiPartition(ngroups)
vmec.verbose = mpi.proc0_world
# print(flush=True)
vmec.mpi = mpi
vmec.fcomm = mpi.comm_groups.py2f()
vmec.iter = mpi.comm_world.bcast(vmec.iter)

comm_grp = mpi.comm_groups
rank = comm_grp.Get_rank()
# vmec       = Vmec("wout_D3D_negtri_pres_scale_10_hres_{0}.nc".format(dof_idx))

isconvrgd = np.array([1], dtype=int)
try:
    vs = vmec_splines(vmec)
except:
    isconvrgd = np.array([0], dtype=int)

print("isconvrgd value", isconvrgd)

if isconvrgd.item() == 1:
    np.save(path1 + "/isconvrgd.npy", isconvrgd)

    if mpi.group == 0:

        ##################################################################################################################
        #####################-------------CALCULATE DIFFERENT PARTS OF THE OBJECTIVE------------##########################
        ##################################################################################################################

        if rank == 0:  # only a single rank in the whole group
            ## Remove the next 6 lines after testing
            ##load target values of the penalty terms
            # aminor0  = 0.682
            # volavgB0 = 0.686
            # aspect0  = 2.42
            # ithresh0 = 0.51
            # P0 = np.save(path1 + "/penalty.npy", np.array([aminor0, volavgB0, aspect0, ithresh0]))

            ## Target values: Presently, they are equal to the starting equilibrium
            T0 = np.load(path1 + "/penalty.npy", allow_pickle=True)

            aminor0, volavgB0, aspect0, ithresh0 = T0

            ntheta1 = int(2 * len(vmec.wout.xm)) + 1
            theta = np.linspace(-1 * np.pi, np.pi, ntheta1)

            helicity_n = -1
            qs_points = np.arange(0, 1.01, 0.1)
            f_qs = QuasisymmetryRatioResidual(
                vmec, qs_points, helicity_m=1, helicity_n=helicity_n
            )
            # print(f_qs.total())
            ## Prefactors in front of the various penalty terms
            # c0, c1, c2, c3, c4, c5 = coeffs
            c0, c1, c2, c3, c4, c5, _, _ = np.load(
                path1 + "/penalty_prefac.npy", allow_pickle=True
            )

            f_minor = (vmec.wout.Aminor_p - aminor0) ** 2
            f_volavgB = (vmec.wout.volavgB - volavgB0) ** 2
            f_aspect = (vmec.wout.aspect - aspect0) ** 2
            # iota should be above ithresh0
            f_iota = (
                np.linalg.norm(
                    np.maximum(-np.abs(vmec.wout.iotas[1:]) + ithresh0, 0)
                    / vmec.wout.ns
                )
            ) ** 2

            if save_dict["eqbm_option"] == 0:  # axisymmetric equilibrium
                f1 = vmec_fieldlines_axisym(
                    vmec_splines(vmec), 0.99, 0.0, theta1d=theta
                )
                Rb = f1.R_1
                Zb = f1.Z_1
                u_ML = np.arctan2(derm(Zb, "l", "o"), derm(Rb, "l", "e"))
                phi_n = np.concatenate(
                    (u_ML[u_ML >= 0] - np.pi / 2, u_ML[u_ML < 0] + 3 * np.pi / 2)
                )
                dl = np.sqrt(derm(Rb, "l", "e") ** 2 + derm(Zb, "l", "o") ** 2)
                R_c = dl / derm(phi_n, "l", "o")
                f_R_c = (
                    np.sum((np.heaviside(R_c + 0.01, 0) - 1).flatten() / ntheta1)
                ) ** 2
            else:
                f_R_c = 0.0

            # f_gamma   =  np.sum(np.maximum(np.load(path1 + "/L{0}.npy".format(int(dof_idx)))[-1], 0))**2
            if int(dof_idx) == 0:
                f = open("penalties.out", "a")
                f.write(
                    f"{iter0}, {f_minor}, {f_volavgB}, {f_aspect}, {f_iota}, {f_R_c}, {f_qs.total()}"
                )
                f.write("\n")
                f.close()

            fobj = np.load(path1 + "/f{0}.npy".format(int(dof_idx)), allow_pickle=True)
            print("dof, fobj", dof_idx, fobj)
            # incomplete objective function
            fobj0_wo_gamma = (
                c0 * f_minor
                + c1 * f_volavgB
                + c2 * f_aspect
                + c3 * f_iota
                + c4 * f_R_c
                + c5 * f_qs.total()
            )

            P0 = np.load(path1 + "/isabs{0}.npy".format(dof_idx), allow_pickle=True)

            if len(np.shape(P0)) != 0:
                P0 = np.vstack((P0, isabs))
                # dfobj = np.vstack((dfobj, dfobj0))
            else:
                P0 = np.append(P0, isabs)
                P0 = np.delete(P0, 0)

            print("isabs = %d" % (isabs))
            np.save(path1 + "/isabs{0}.npy".format(dof_idx), P0)

            # insert the Dof value at the beginning of p00
            # prxy0_gthrd = np.insert(prxy0_gthrd, 0, dof_idx)
            if len(np.shape(fobj)) != 0:
                fobj = np.vstack((fobj, fobj0_wo_gamma))
                # dfobj = np.vstack((dfobj, dfobj0))
            else:
                fobj = np.append(fobj, fobj0_wo_gamma)
                fobj = np.delete(fobj, 0)

            print("dof, fobj", dof_idx, fobj)

            np.save(path1 + "/f{0}.npy".format(int(dof_idx)), fobj)

            spr.call(["rm -r dcon*.txt\n rm -r threed1.*"], shell=True)


else:
    np.save(path1 + "/isconvrgd.npy", isconvrgd)
    if mpi.group == 0:
        if rank == 0:  # only a single rank in the whole group
            ## Prefactors in front of the various penalty terms
            f_minor = 9.99
            f_volavgB = 9.99
            f_aspect = 9.99
            f_iota = 9.99
            f_R_c = 9.99
            f_qs = 9.99
            if int(dof_idx) == 0:
                f = open("penalties.out", "a")
                f.write(
                    f"{iter0}, {f_minor}, {f_volavgB}, {f_aspect}, {f_iota}, {f_R_c}, {f_qs}"
                )
                f.write("\n")
                f.close()

            fobj = np.load(path1 + "/f{0}.npy".format(int(dof_idx)), allow_pickle=True)
            print("dof, fobj", dof_idx, fobj)
            # incomplete objective function
            fobj0_wo_gamma = 99

            P0 = np.load(path1 + "/isabs{0}.npy".format(dof_idx), allow_pickle=True)

            if len(np.shape(P0)) != 0:
                P0 = np.vstack((P0, isabs))
                # dfobj = np.vstack((dfobj, dfobj0))
            else:
                P0 = np.append(P0, isabs)
                P0 = np.delete(P0, 0)

            print("isabs = %d" % (isabs))
            np.save(path1 + "/isabs{0}.npy".format(dof_idx), P0)

            # insert the Dof value at the beginning of p00
            # prxy0_gthrd = np.insert(prxy0_gthrd, 0, dof_idx)
            if len(np.shape(fobj)) != 0:
                fobj = np.vstack((fobj, fobj0_wo_gamma))
                # dfobj = np.vstack((dfobj, dfobj0))
            else:
                fobj = np.append(fobj, fobj0_wo_gamma)
                fobj = np.delete(fobj, 0)

            print("dof, fobj", dof_idx, fobj)

            np.save(path1 + "/f{0}.npy".format(int(dof_idx)), fobj)

            spr.call(["rm -r dcon*.txt\n rm -r threed1.*"], shell=True)
