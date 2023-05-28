#!/bin/bash

#SBATCH --qos=regular
#SBATCH --time=00:08:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=96
#SBATCH --cpus-per-task=1


export OMP_NUM_THREADS=1;\
#export NUMEXPR_MAX_THREADS=1;\


##conda deactivate;
#module purge;
#module load openmpi/gcc/4.1.0; 
#module load hdf5/gcc/1.10.6;
#module load netcdf/gcc/hdf5-1.10.6/4.7.4
#module load intel-mkl/2021.1.1;
#module load anaconda3/2021.5;
#conda activate sims_gcc_py39;








PMIX_MCA_psec=native PMIX_MCA_gds=hash srun -N 1 -n13 -c 1 --mpi=pmix_v3 singularity run simsopt_v0.13.0.sif /venv/bin/python -u ball_scan.py 1 0 13 & 
 PMIX_MCA_psec=native PMIX_MCA_gds=hash srun -N 1 -n13 -c 1 --mpi=pmix_v3 singularity run simsopt_v0.13.0.sif /venv/bin/python -u ball_scan.py 1 1 13 & 
 PMIX_MCA_psec=native PMIX_MCA_gds=hash srun -N 1 -n13 -c 1 --mpi=pmix_v3 singularity run simsopt_v0.13.0.sif /venv/bin/python -u ball_scan.py 1 2 13 & 
 PMIX_MCA_psec=native PMIX_MCA_gds=hash srun -N 1 -n13 -c 1 --mpi=pmix_v3 singularity run simsopt_v0.13.0.sif /venv/bin/python -u ball_scan.py 1 3 13 & 
 PMIX_MCA_psec=native PMIX_MCA_gds=hash srun -N 1 -n13 -c 1 --mpi=pmix_v3 singularity run simsopt_v0.13.0.sif /venv/bin/python -u ball_scan.py 1 4 13 & 
 PMIX_MCA_psec=native PMIX_MCA_gds=hash srun -N 1 -n13 -c 1 --mpi=pmix_v3 singularity run simsopt_v0.13.0.sif /venv/bin/python -u ball_scan.py 1 5 13 & 
 PMIX_MCA_psec=native PMIX_MCA_gds=hash srun -N 1 -n13 -c 1 --mpi=pmix_v3 singularity run simsopt_v0.13.0.sif /venv/bin/python -u ball_scan.py 1 6 13 & 
 wait 
 
