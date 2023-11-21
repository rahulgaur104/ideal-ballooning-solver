#!/usr/bin/env python
"""
This script runs at every iteration and resets the done_arr and cntr_arr. These arrays are used to redistribute processors in the fine-grained GS2 runs, i.e., load balancing
"""

import numpy as np
import os
import subprocess as spr
import pickle
import sys

path0 = os.getcwd() + "/GS2_files"
path1 = os.getcwd() + "/save_n_load"

reset_type = sys.argv[1]

with open("params_dict.pkl", "rb") as f:
    save_dict = pickle.load(f)

totalndofs = save_dict["totalndofs"]
nsurfs = save_dict["nsurfs"]
# The user needs to have prior information about the optimization runs
# before creating done_arr and cntr_arr
# njobsGS2fl  = save_dict['njobsGS2fl']
# execsperjob = int(save_dict['totalnexecGS2fl']/njobsGS2fl)

# if reset_type == 'f': # full reset
#    for i in range(njobsGS2fl):
#        np.save(path1 + "/done{0}.npy".format(int(i)), np.linspace(i*execsperjob, (i+1)*execsperjob-1, execsperjob, dtype=int))
#        np.save(path1 + "/cntr_arr{0}.npy".format(int(i)), np.ones((execsperjob,), dtype=int))
#
#    # This doesn't need to be reset now as we have fixed nlambda
#    for i in range(int((totalndofs+1)*nsurfs)):
#        np.save(path1 + "/cntr_allowd{0}.npy".format(int(i)), np.empty([], dtype=int))
# else:
#    for i in range(njobsGS2fl):
#        np.save(path1 + "/done{0}.npy".format(int(i)), np.linspace(i*execsperjob, (i+1)*execsperjob-1, execsperjob, dtype=int))
#        np.save(path1 + "/cntr_arr{0}.npy".format(int(i)), np.ones((execsperjob,), dtype=int))
#
