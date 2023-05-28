#!/bin/bash

#SBATCH --qos=regular
#SBATCH --time=00:08:00
#SBATCH --nodes=4
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








PMIX_MCA_psec=native PMIX_MCA_gds=hash srun -N 4 -n10 -c 1 --mpi=pmix_v3 singularity run simsopt_v0.13.0.sif /venv/bin/python -u ball_scan.py 0 36 5 & 
 PMIX_MCA_psec=native PMIX_MCA_gds=hash srun -N 4 -n10 -c 1 --mpi=pmix_v3 singularity run simsopt_v0.13.0.sif /venv/bin/python -u ball_scan.py 0 37 5 & 
 PMIX_MCA_psec=native PMIX_MCA_gds=hash srun -N 4 -n10 -c 1 --mpi=pmix_v3 singularity run simsopt_v0.13.0.sif /venv/bin/python -u ball_scan.py 0 38 5 & 
 PMIX_MCA_psec=native PMIX_MCA_gds=hash srun -N 4 -n10 -c 1 --mpi=pmix_v3 singularity run simsopt_v0.13.0.sif /venv/bin/python -u ball_scan.py 0 39 5 & 
 PMIX_MCA_psec=native PMIX_MCA_gds=hash srun -N 4 -n10 -c 1 --mpi=pmix_v3 singularity run simsopt_v0.13.0.sif /venv/bin/python -u ball_scan.py 0 40 5 & 
 PMIX_MCA_psec=native PMIX_MCA_gds=hash srun -N 4 -n10 -c 1 --mpi=pmix_v3 singularity run simsopt_v0.13.0.sif /venv/bin/python -u ball_scan.py 0 41 5 & 
 PMIX_MCA_psec=native PMIX_MCA_gds=hash srun -N 4 -n10 -c 1 --mpi=pmix_v3 singularity run simsopt_v0.13.0.sif /venv/bin/python -u ball_scan.py 0 42 5 & 
 PMIX_MCA_psec=native PMIX_MCA_gds=hash srun -N 4 -n10 -c 1 --mpi=pmix_v3 singularity run simsopt_v0.13.0.sif /venv/bin/python -u ball_scan.py 0 43 5 & 
 PMIX_MCA_psec=native PMIX_MCA_gds=hash srun -N 4 -n10 -c 1 --mpi=pmix_v3 singularity run simsopt_v0.13.0.sif /venv/bin/python -u ball_scan.py 0 44 5 & 
 PMIX_MCA_psec=native PMIX_MCA_gds=hash srun -N 4 -n10 -c 1 --mpi=pmix_v3 singularity run simsopt_v0.13.0.sif /venv/bin/python -u ball_scan.py 0 45 5 & 
 PMIX_MCA_psec=native PMIX_MCA_gds=hash srun -N 4 -n10 -c 1 --mpi=pmix_v3 singularity run simsopt_v0.13.0.sif /venv/bin/python -u ball_scan.py 0 46 5 & 
 PMIX_MCA_psec=native PMIX_MCA_gds=hash srun -N 4 -n10 -c 1 --mpi=pmix_v3 singularity run simsopt_v0.13.0.sif /venv/bin/python -u ball_scan.py 0 47 5 & 
 PMIX_MCA_psec=native PMIX_MCA_gds=hash srun -N 4 -n10 -c 1 --mpi=pmix_v3 singularity run simsopt_v0.13.0.sif /venv/bin/python -u ball_scan.py 0 48 5 & 
 PMIX_MCA_psec=native PMIX_MCA_gds=hash srun -N 4 -n10 -c 1 --mpi=pmix_v3 singularity run simsopt_v0.13.0.sif /venv/bin/python -u ball_scan.py 0 49 5 & 
 PMIX_MCA_psec=native PMIX_MCA_gds=hash srun -N 4 -n10 -c 1 --mpi=pmix_v3 singularity run simsopt_v0.13.0.sif /venv/bin/python -u ball_scan.py 0 50 5 & 
 PMIX_MCA_psec=native PMIX_MCA_gds=hash srun -N 4 -n10 -c 1 --mpi=pmix_v3 singularity run simsopt_v0.13.0.sif /venv/bin/python -u ball_scan.py 0 51 5 & 
 PMIX_MCA_psec=native PMIX_MCA_gds=hash srun -N 4 -n10 -c 1 --mpi=pmix_v3 singularity run simsopt_v0.13.0.sif /venv/bin/python -u ball_scan.py 0 52 5 & 
 PMIX_MCA_psec=native PMIX_MCA_gds=hash srun -N 4 -n10 -c 1 --mpi=pmix_v3 singularity run simsopt_v0.13.0.sif /venv/bin/python -u ball_scan.py 0 53 5 & 
 PMIX_MCA_psec=native PMIX_MCA_gds=hash srun -N 4 -n10 -c 1 --mpi=pmix_v3 singularity run simsopt_v0.13.0.sif /venv/bin/python -u ball_scan.py 0 54 5 & 
 PMIX_MCA_psec=native PMIX_MCA_gds=hash srun -N 4 -n10 -c 1 --mpi=pmix_v3 singularity run simsopt_v0.13.0.sif /venv/bin/python -u ball_scan.py 0 55 5 & 
 PMIX_MCA_psec=native PMIX_MCA_gds=hash srun -N 4 -n10 -c 1 --mpi=pmix_v3 singularity run simsopt_v0.13.0.sif /venv/bin/python -u ball_scan.py 0 56 5 & 
 PMIX_MCA_psec=native PMIX_MCA_gds=hash srun -N 4 -n10 -c 1 --mpi=pmix_v3 singularity run simsopt_v0.13.0.sif /venv/bin/python -u ball_scan.py 0 57 5 & 
 PMIX_MCA_psec=native PMIX_MCA_gds=hash srun -N 4 -n10 -c 1 --mpi=pmix_v3 singularity run simsopt_v0.13.0.sif /venv/bin/python -u ball_scan.py 0 58 5 & 
 PMIX_MCA_psec=native PMIX_MCA_gds=hash srun -N 4 -n10 -c 1 --mpi=pmix_v3 singularity run simsopt_v0.13.0.sif /venv/bin/python -u ball_scan.py 0 59 5 & 
 PMIX_MCA_psec=native PMIX_MCA_gds=hash srun -N 4 -n10 -c 1 --mpi=pmix_v3 singularity run simsopt_v0.13.0.sif /venv/bin/python -u ball_scan.py 0 60 5 & 
 PMIX_MCA_psec=native PMIX_MCA_gds=hash srun -N 4 -n10 -c 1 --mpi=pmix_v3 singularity run simsopt_v0.13.0.sif /venv/bin/python -u ball_scan.py 0 61 5 & 
 PMIX_MCA_psec=native PMIX_MCA_gds=hash srun -N 4 -n10 -c 1 --mpi=pmix_v3 singularity run simsopt_v0.13.0.sif /venv/bin/python -u ball_scan.py 0 62 5 & 
 PMIX_MCA_psec=native PMIX_MCA_gds=hash srun -N 4 -n10 -c 1 --mpi=pmix_v3 singularity run simsopt_v0.13.0.sif /venv/bin/python -u ball_scan.py 0 63 5 & 
 PMIX_MCA_psec=native PMIX_MCA_gds=hash srun -N 4 -n10 -c 1 --mpi=pmix_v3 singularity run simsopt_v0.13.0.sif /venv/bin/python -u ball_scan.py 0 64 5 & 
 PMIX_MCA_psec=native PMIX_MCA_gds=hash srun -N 4 -n10 -c 1 --mpi=pmix_v3 singularity run simsopt_v0.13.0.sif /venv/bin/python -u ball_scan.py 0 65 5 & 
 PMIX_MCA_psec=native PMIX_MCA_gds=hash srun -N 4 -n10 -c 1 --mpi=pmix_v3 singularity run simsopt_v0.13.0.sif /venv/bin/python -u ball_scan.py 0 66 5 & 
 PMIX_MCA_psec=native PMIX_MCA_gds=hash srun -N 4 -n10 -c 1 --mpi=pmix_v3 singularity run simsopt_v0.13.0.sif /venv/bin/python -u ball_scan.py 0 67 5 & 
 PMIX_MCA_psec=native PMIX_MCA_gds=hash srun -N 4 -n10 -c 1 --mpi=pmix_v3 singularity run simsopt_v0.13.0.sif /venv/bin/python -u ball_scan.py 0 68 5 & 
 PMIX_MCA_psec=native PMIX_MCA_gds=hash srun -N 4 -n10 -c 1 --mpi=pmix_v3 singularity run simsopt_v0.13.0.sif /venv/bin/python -u ball_scan.py 0 69 5 & 
 PMIX_MCA_psec=native PMIX_MCA_gds=hash srun -N 4 -n10 -c 1 --mpi=pmix_v3 singularity run simsopt_v0.13.0.sif /venv/bin/python -u ball_scan.py 0 70 5 & 
 PMIX_MCA_psec=native PMIX_MCA_gds=hash srun -N 4 -n10 -c 1 --mpi=pmix_v3 singularity run simsopt_v0.13.0.sif /venv/bin/python -u ball_scan.py 0 71 5 & 
 PMIX_MCA_psec=native PMIX_MCA_gds=hash srun -N 4 -n10 -c 1 --mpi=pmix_v3 singularity run simsopt_v0.13.0.sif /venv/bin/python -u ball_scan.py 0 72 5 & 
 wait 
 
