#!/usr/bin/env python
"""
This script is only run one at the beginning of an optimization loop
"""

import subprocess as spr
import os
import time
import pdb



#pdb.set_trace()
# The next 35 lines check for the status of the submitted slurm file
fname1 = 'sbatchout_set_x0.txt'
fname2 = 'slurmstatus_set_x0.txt'

rmcmd = 'rm ' + os.getcwd() +  '/' + fname1 + ' ' + os.getcwd()  + '/' + fname2
p = spr.Popen(rmcmd, shell=True)
p.wait()

with open(fname1, 'w') as myoutput:
	slurmcmd = 'sbatch ' + os.getcwd() +  '/slurm_x0.sl'
	p = spr.Popen(slurmcmd, shell=True, stdout = myoutput)
	p.wait()

with open(fname1, 'r') as myoutput:
	lines = myoutput.readlines()

jobid = int(eval(lines[0].split(' ')[3]))

with open(fname2, 'w') as myoutput:
	p = spr.Popen('squeue -j {0}'.format(jobid), shell=True, stdout=myoutput)
	p.wait()

#while len(open(fname2, 'r').readlines()) == 2:
#	time.sleep(2)
#	#print("sleeping!")
#	rmcmd = 'rm ' + os.getcwd()  + '/' + fname2
#	p = spr.Popen(rmcmd, shell=True)
#	p.wait()
#	with open(fname2, 'w') as myoutput:
#		p = spr.Popen('squeue -j {0}'.format(jobid), shell=True, stdout=myoutput)
#		p.wait()
#	#spr.call(['tail -f slurm-{0}.out'.format(jobid)], shell=True)

while ' '.join(open(fname2, 'r').readlines()[1].split()).split(' ')[4] != 'CG':
	time.sleep(2)
	#print("sleeping!")
	rmcmd = 'rm ' + os.getcwd()  + '/' + fname2
	p = spr.Popen(rmcmd, shell=True)
	p.wait()
	with open(fname2, 'w') as myoutput:
		p = spr.Popen('squeue -j {0}'.format(jobid), shell=True, stdout=myoutput)
		p.wait()
	#spr.call(['tail -f slurm-{0}.out'.format(jobid)], shell=True)

time.sleep(5)

