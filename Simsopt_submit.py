#!/usr/bin/env python
"""
This script submits the batch file used to calculate the geometry for all the Dofs and surfaces.
"""

import subprocess as spr
import os
import time
import sys
import pdb
import pickle

iter0 = int(eval(sys.argv[1]))

with open("params_dict.pkl", "rb") as f:
    save_dict = pickle.load(f)

totalndofs = save_dict["totalndofs"]
nsurfs = save_dict["nsurfs"]

nprocspernode = save_dict["nprocspernode"]

nodes = save_dict["nodespersimsopt"]

ngroups = int(nodes * nprocspernode / ((totalndofs + 1) * nsurfs))
nprocs_simsopt = int(nodes * nprocspernode / (totalndofs + 1))

spr.call(
    ["sed -ri 's@nodes=[0-9]*@nodes={0:d}@g' slurm_Simsopt_template.sl".format(nodes)],
    shell=True,
)

slurmtemplatecpcmd = "cp -r slurm_Simsopt_template.sl slurm_Simsopt.sl"
p = spr.Popen(slurmtemplatecpcmd, shell=True, stdout=spr.PIPE, stdin=spr.PIPE)
p.wait()

with open("slurm_Simsopt.sl", "a") as f:
    # lines = f.readlines()
    for i in range(totalndofs + 1):
        Simsoptcmd2 = "PMIX_MCA_psec=native  PMIX_MCA_gds=hash srun -N {0} -n{1} -c 1 --mpi=pmix_v3 singularity run simsopt_v0.13.0.sif /venv/bin/python -u Simsopt_runner.py {2} {3} {4} & \n".format(
            nodes, nprocs_simsopt, iter0, i, ngroups
        )
        # If you are using a SIMSOPT Python environment uncomment the line below
        # Simsoptcmd2  = "srun -N {0} -n{1} -c 1 python3 -u Simsopt_runner.py {2} {3} {4} & \n".format(nodes, nprocs_simsopt, iter0, i, ngroups)
        f.write(Simsoptcmd2)

    f.write("wait \n")

# pdb.set_trace()
# The next 35 lines check for the status of the submitted slurm file
fname1 = "sbatchout_Simsopt.txt"
fname2 = "slurmstatus_Simsopt.txt"

rmcmd = "rm " + os.getcwd() + "/" + fname1 + " " + os.getcwd() + "/" + fname2
p = spr.Popen(rmcmd, shell=True)
p.wait()

with open(fname1, "w") as myoutput:
    slurmcmd = "sbatch " + os.getcwd() + "/slurm_Simsopt.sl"
    p = spr.Popen(slurmcmd, shell=True, stdout=myoutput)
    p.wait()

with open(fname1, "r") as myoutput:
    lines = myoutput.readlines()

jobid = int(eval(lines[0].split(" ")[3]))

with open(fname2, "w") as myoutput:
    p = spr.Popen("squeue -j {0}".format(jobid), shell=True, stdout=myoutput)
    p.wait()


while " ".join(open(fname2, "r").readlines()[1].split()).split(" ")[4] != "CG":
    time.sleep(2)
    # print("sleeping!")
    rmcmd = "rm " + os.getcwd() + "/" + fname2
    p = spr.Popen(rmcmd, shell=True)
    p.wait()
    with open(fname2, "w") as myoutput:
        p = spr.Popen("squeue -j {0}".format(jobid), shell=True, stdout=myoutput)
        p.wait()


time.sleep(10)
