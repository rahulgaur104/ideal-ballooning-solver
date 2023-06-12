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








