#!/bin/bash

#SBATCH --qos=debug
#SBATCH --time=00:04:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1



PMIX_MCA_psec=native srun -n 1 --mpi=pmix_v3 singularity run simsopt_v0.13.0.sif /venv/bin/python set_x0.py 

## If you are using a Python environment, uncomment the line below 
#srun -n 1 python3 -u set_x0.py 

