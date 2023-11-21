#!/usr/bin/env python
"""
This script creates empty arrays using during the code for difference purposes such as saving the max growth rate information, proxies etc.
"""

import numpy as np
import os
import subprocess as spr
import pickle


with open("params_dict.pkl", "rb") as f:
    save_dict = pickle.load(f)

totalndofs = save_dict["totalndofs"]
nsurfs = save_dict["nsurfs"]

path0 = os.getcwd() + "/GS2_files"
path1 = os.getcwd() + "/save_n_load"
# Removing files from previous optimization and creating new files


rmold = "rm -r df0_list.out \n rm -r f0_list.out \n rm penalties.out \n rm -r {0}/grid_files/* \n rm *_list.out \n rm -r {0}/gs2* \n rm -r {0}/.gs2* \n rm -r {0}/rs* \n rm -r {1}/*.npy \n rm -r input.dof* \n rm -r dcon* \n rm -r jxbout_* \n rm -r threed1.* \n rm slurm-*.out \n rm wout_dof*.nc".format(
    path0, path1
)

# print(rmold)
# args = shlex.split(rmold)
# p    = spr.Popen(rmold, shell=True, stdout=spr.PIPE, stdin=spr.PIPE)
# p.wait()
# print(p.stdout.read())

spr.call([rmold], shell=True)

eqbm_arr = ["D3D", "NCSX", "HBERG", "NW"]
eqbm_option = save_dict["eqbm_option"]  # to decide if it's a tokamak or stellarator

VMECinpcp = ""
for i in range(totalndofs + 1):
    VMECcpcmd = (
        "cp -r input.template_" + eqbm_arr[eqbm_option] + " input.dof{0} \n".format(i)
    )
    VMECinpcp = VMECinpcp + VMECcpcmd

# args = shlex.split(VMECinpcp)
p = spr.Popen(VMECinpcp, shell=True, stdout=spr.PIPE, stdin=spr.PIPE)
p.wait()
print(p.stdout.read())


np.save(path1 + "/isconvrgd.npy", np.ones((1,), dtype=int))


# for i in range(totalndofs+1):
#    GS2inpcp = ""
#    for j in range(nsurfs):
#        GS2cpcmd = "cp -r gs2_template.in {0}/gs2_dof{1}_rhoidx{2}.in \n ".format(path0, i, j)
#        GS2inpcp = GS2inpcp + GS2cpcmd
#
#    #args = shlex.split(GS2inpcp)
#    p    = spr.Popen(GS2inpcp, shell=True, stdout=spr.PIPE, stdin=spr.PIPE)
#    p.wait()
#    print(p.stdout.read())


for i in range(nsurfs):
    np.save(path1 + "/iscoarsenan{0:d}.npy".format(i), np.zeros((1,), dtype=int))


# 0, or 1: 0 if we take an absolute step, 1 if we take a relative step
for i in range((totalndofs + 1)):
    np.save(path1 + "/isabs{0:d}.npy".format(i), np.empty([]))

# Overall state vector
for i in range((totalndofs + 1)):
    np.save(path1 + "/x{0:d}.npy".format(i), np.empty([]))

# Overall objective function
for i in range((totalndofs + 1)):
    np.save(path1 + "/f{0:d}.npy".format(i), np.empty([]))

# Derivative of the objective function
for i in range((totalndofs + 1)):
    np.save(path1 + "/df{0:d}.npy".format(i), np.empty([]))


# ballooning gamma_max; Dof number of empty aRRAYS, (ns, niter)
for i in range((totalndofs + 1)):
    np.save(path1 + "/ball_gam{0:d}.npy".format(i), np.empty([]))

# ballooning theta0 at gamma_max; Dof number of empty arrays, (ns, niter)
for i in range((totalndofs + 1)):
    np.save(path1 + "/ball_theta0{0:d}.npy".format(i), np.empty([]))

# ballooning theta0 at gamma_max; Dof number of empty arrays, (ns, niter)
for i in range((totalndofs + 1)):
    np.save(path1 + "/ball_alpha{0:d}.npy".format(i), np.empty([]))

# \lambda_max; Dof number of empty arrays, (ns, niter)
for i in range((totalndofs + 1)):
    np.save(path1 + "/L{0:d}.npy".format(i), np.empty([]))

# ky at max lambda; Dof number of empty arrays, (1, )
for i in range(totalndofs + 1):
    np.save(path1 + "/ky{0:d}.npy".format(i), np.empty([]))

# kx temporary array for each iteration after fine-graining
for i in range(totalndofs + 1):
    np.save(path1 + "/kx{0:d}.npy".format(i), np.empty([]))


# lambda after the coarse grid scan for each iteration(before fine-graining the max)
# for each dof and each flux surface
for i in range((totalndofs + 1) * nsurfs):
    np.save(path1 + "/Ltc{0:d}.npy".format(i), np.empty([]))

# omega after the coarse grid scan for each iteration(before fine-graining the max)
# for each dof and each flux surface
for i in range((totalndofs + 1) * nsurfs):
    np.save(path1 + "/omtc{0:d}.npy".format(i), np.empty([]))

# temporary ky arrays after the coarse scan
for i in range((totalndofs + 1) * nsurfs):
    np.save(path1 + "/kytc{0:d}.npy".format(i), np.empty([]))

# temporary kx arrays after the coarse scan
for i in range((totalndofs + 1) * nsurfs):
    np.save(path1 + "/kxtc{0:d}.npy".format(i), np.empty([]))


# lambda temporary for each iteration(after finding the maximum)
# for each dof and each flux surface
for i in range((totalndofs + 1) * nsurfs):
    np.save(path1 + "/Lt{0:d}.npy".format(i), np.empty([]))

# omega temporary for each iteration(after finding the maximum)
# for each dof and each flux surface
for i in range((totalndofs + 1) * nsurfs):
    np.save(path1 + "/omt{0:d}.npy".format(i), np.empty([]))

# ky at max lambda; Dof number of empty arrays, (1, )
for i in range((totalndofs + 1) * nsurfs):
    np.save(path1 + "/kyt{0:d}.npy".format(i), np.empty([]))

# kx temporary array for each iteration after fine-graining
for i in range((totalndofs + 1) * nsurfs):
    np.save(path1 + "/kxt{0:d}.npy".format(i), np.empty([]))


# Proxy zero
for i in range((totalndofs + 1) * nsurfs):
    np.save(path1 + "/P{0:d}0.npy".format(i), np.empty([]))

# Proxy one; dimensions (ns, niter)

for i in range((totalndofs + 1) * nsurfs):
    np.save(path1 + "/P{0:d}1.npy".format(i), np.empty([]))
