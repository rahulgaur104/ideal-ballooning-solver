#!/usr/bin/env python
"""
This script submits the batch file used to calculate the geometry for all the Dofs and surfaces.
"""

import numpy as np
import subprocess as spr
import os
import time
import sys
import pdb
import pickle

iter0  = int(eval(sys.argv[1]))


with open('params_dict.pkl', 'rb') as f:
    save_dict = pickle.load(f)

totalndofs = save_dict['totalndofs']
nsurfs     = save_dict['nsurfs']

nprocspernode = save_dict['nprocspernode']   
numjobs = save_dict['njobsball'] # numjobs number of separate job files

totalexecs = save_dict['totalnexecball'] # totalndofs + 1

execsperjob = int(totalexecs/numjobs) # number of python3 executables per slurm job

nodes  = save_dict['nodesperball'] # total number of nodes used
nodesperjob = int(nodes/numjobs)

# we want ball_scan on nsurfs. Everything else is parallelized
#ngroups = execsperjob/nsurfs
#ngroups = nsurfs
ngroups = int(nodesperjob*nprocspernode/totalexecs)
nsurfs  = int(ngroups) 

assert nodesperjob > 0, "Error in ball_submit! nodesperjob = 0. Use more nodes"

nprocs = int((nodes * nprocspernode)/totalexecs) # gives us nprocessors per ball_scan.py instance
# We need a nprocs that is perfectly divisible by nsurfs
# That way for each surface, we can have an integer number of alpha
while np.mod(nprocs, nsurfs) != 0:
    nprocs = nprocs - 1

jobid_arr = np.zeros((numjobs, ))

spr.call(["sed -ri 's@nodes=[0-9]*@nodes={0:d}@g'  slurm_ball_scan_template.sl".format(nodesperjob)], shell=True)


# create a list of all the python executables
execlist = []

for i in range(totalexecs): # Each dof will have a ball_submit.py script that calculates growth rates on all surfs
    ballcmd2  = "PMIX_MCA_psec=native PMIX_MCA_gds=hash srun -N {0} -n{1} -c 1 --mpi=pmix_v3 singularity run simsopt_v0.13.0.sif /venv/bin/python -u ball_scan.py {2} {3} {4} & \n".format(nodesperjob, nprocs, iter0, i, ngroups)
    #ballcmd2  = "PMIX_MCA_psec=native srun -N {0} -n{1} -c 1 --mpi=pmix_v3 singularity run simsopt_v0.11.0.sif /venv/bin/python -u ball_scan.py {2} {3} {4} & \n".format(nodesperjob, nprocs, iter0, i, ngroups)
    # If you are using a SIMSOPT Python environment uncomment the line below
    #ballcmd2  = "srun -N {0} -n{1} -c 1 python3 -u ball_scan.py {2} {3} {4} & \n".format(nodesperjob, nprocs, iter0, i, ngroups)
    execlist.append(ballcmd2)

# The total number of execs is almost never divisible by numjobs
# So we split the executables equally and move the rest to the last slurm job
rem_execs = np.mod(totalexecs, numjobs)    
#print(rem_execs, totalexecs, numjobs)


if rem_execs == 0: 
    for k in range(numjobs):
        slurmtemplatecpcmd = "cp -r slurm_ball_scan_template.sl slurm_ball_scan{0:d}.sl".format(k)
        p    = spr.Popen(slurmtemplatecpcmd, shell=True, stdout=spr.PIPE, stdin=spr.PIPE)
        p.wait()
        
        with open('slurm_ball_scan{0}.sl'.format(k), 'a') as f: #add the python execs to the slurm script
        	    f.write(' '.join(execlist[k*execsperjob:(k+1)*execsperjob]) + " wait \n \n")

        # The next 35 lines check the status of the submitted slurm file
        fname1 = 'sbatchout_ball_scan{0}.txt'.format(k)
        fname2 = 'slurmstatus_ball_scan{0}.txt'.format(k)
        
        rmcmd = 'rm ' + os.getcwd() +  '/' + fname1 + ' ' + os.getcwd()  + '/' + fname2
        p = spr.Popen(rmcmd, shell=True)
        p.wait()
 

        with open(fname1, 'w') as myoutput: #submit the slurm job file
        	slurmcmd = 'sbatch ' + os.getcwd() +  '/slurm_ball_scan{0:d}.sl'.format(k)
        	p = spr.Popen(slurmcmd, shell=True, stdout = myoutput)
        	p.wait()
        
        with open(fname1, 'r') as myoutput: # get the jobid of the submitted job
        	lines = myoutput.readlines()
        
        jobid_arr[k] = int(eval(lines[0].split(' ')[3]))
        
        with open(fname2, 'w') as myoutput:
        	p = spr.Popen('squeue -j {0}'.format(jobid_arr[k]), shell=True, stdout=myoutput)
        	p.wait()

else:
    for k in range(numjobs):
        if k == numjobs-1:
            slurmtemplatecpcmd = "cp -r slurm_ball_scan_template.sl slurm_ball_scan{0:d}.sl".format(k)
            p    = spr.Popen(slurmtemplatecpcmd, shell=True, stdout=spr.PIPE, stdin=spr.PIPE)
            p.wait()
            with open('slurm_ball_scan{0}.sl'.format(k), 'a') as f: #add the python execs to the slurm script
                f.write(' '.join(execlist[k*execsperjob:]) + " wait \n \n")
        else:
            slurmtemplatecpcmd = "cp -r slurm_ball_scan_template.sl slurm_ball_scan{0:d}.sl".format(k)
            p    = spr.Popen(slurmtemplatecpcmd, shell=True, stdout=spr.PIPE, stdin=spr.PIPE)
            p.wait()
            with open('slurm_ball_scan{0}.sl'.format(k), 'a') as f: #add the python execs to the slurm script
    	        f.write(' '.join(execlist[k*execsperjob:(k+1)*execsperjob]) + " wait \n \n")

        # The next 35 lines check the status of the submitted slurm file
        fname1 = 'sbatchout_ball_scan{0}.txt'.format(k)
        fname2 = 'slurmstatus_ball_scan{0}.txt'.format(k)
        
        rmcmd = 'rm ' + os.getcwd() +  '/' + fname1 + ' ' + os.getcwd()  + '/' + fname2
        p = spr.Popen(rmcmd, shell=True)
        p.wait()
 

        with open(fname1, 'w') as myoutput: #submit the slurm job file
        	slurmcmd = 'sbatch ' + os.getcwd() +  '/slurm_ball_scan{0:d}.sl'.format(k)
        	p = spr.Popen(slurmcmd, shell=True, stdout = myoutput)
        	p.wait()
        
        with open(fname1, 'r') as myoutput: # get the jobid of the submitted job
        	lines = myoutput.readlines()
        
        jobid_arr[k] = int(eval(lines[0].split(' ')[3]))
        
        with open(fname2, 'w') as myoutput:
        	p = spr.Popen('squeue -j {0}'.format(jobid_arr[k]), shell=True, stdout=myoutput)
        	p.wait()


while len(open('slurmstatus_ball_scan0.txt', 'r').readlines()) > 1:
    time.sleep(2)
    fname2 = 'slurmstatus_ball_scan0.txt'
    rmcmd = 'rm ' + os.getcwd()  + '/' + fname2
    p = spr.Popen(rmcmd, shell=True)
    p.wait()
    with open(fname2, 'w') as myoutput:
    	p = spr.Popen('squeue -u {0}'.format(save_dict['username']), shell=True, stdout=myoutput)
    	p.wait()

## This can still throw an error if the unrelated job gets cancelled before 
## this one
#while ' '.join(open(fname2, 'r').readlines()[1].split()).split(' ')[4] != 'CG':
#	time.sleep(2)
#	#print("sleeping!")
#	rmcmd = 'rm ' + os.getcwd()  + '/' + fname2
#	p = spr.Popen(rmcmd, shell=True)
#	p.wait()
#	with open(fname2, 'w') as myoutput:
#		p = spr.Popen('squeue -j {0}'.format(jobid), shell=True, stdout=myoutput)
#		p.wait()
#			
#	#spr.call(['tail -f slurm-{0}.out'.format(jobid)], shell=True)
#
#time.sleep(10)


