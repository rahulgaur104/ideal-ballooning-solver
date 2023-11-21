#!/usr/bin/env python
"""
This script is only run once; at the beginning to obtian and set the starting x0
"""
import numpy as np
import subprocess as spr
from simsopt.mhd.vmec import Vmec
from simsopt.mhd.profiles import (
    Profile,
    ProfilePolynomial,
    ProfileScaled,
    ProfileSpline,
)
import pickle
import pdb
import os

vmec = Vmec("input.dof0", keep_all_files=False)

with open("params_dict.pkl", "rb") as f:
    save_dict = pickle.load(f)

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
    vmec._boundary.fix_all()
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


x0 = vmec.x

path0 = os.getcwd() + "/save_n_load"
L0 = np.load(path0 + "/x0.npy", allow_pickle=True)

L0 = np.append(L0, x0)
L0 = np.delete(L0, 0)

np.save(path0 + "/x0.npy", L0)
