# ballopt: stellarator and tokamak stability optimizer

This repository contains a set of Python scripts that, along with the SIMSOPT package can optimize or evaluate a tokamak or stellarator equilibrium against the infinite-n, ideal ballooning mode using an adjoint-based method.

The details of implementation of this method and the results are available on arxiv: [An adjoint-based method to optimize MHD equilibria against the infinite-n ideal ballooning mode](https://arxiv.org/abs/2302.07673)

## Dependencies

If you are using a cluster or supercomputer that uses SLURM, you need a Python version >= 3.8 and a [SIMSOPT singularity image](https://simsopt.readthedocs.io/en/latest/containers.html#singularity-container) to run this optimizer. However, if you are running this on a laptop or a device without SLURM, you need to install the SIMSOPT optimization package and the VMEC2000 code from the [SIMSOPT GitHub page] (https://github.com/hiddenSymmetries/simsopt)

## Instructions to reproduce the results in the adjoint-ballooning paper

The equilibria studied in the paper above are presented in this repository. You should be able to optimize any of the three equilibria described below:

0. D3D
1. NCSX
2. HBERG

First choose the eqbm\_option in create\_dict.py. For DIII-D it's 0, otherwise choose 1 or 2. Next, run

python sims\_runner\_\<equilibrium\_name\>.py

from your login node. After each iteration, the data are saved in the directory save\_n\_load.

Please do not hesitate to contact me at rgaur@terpmail.umd.edu if you need any help running or understanding this optimizer.

## Relevant papers
* [COBRA: An Optimized Code for Fast Analysis of Ideal Ballooning Stability of Three-Dimensional Magnetic Equilibria](https://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.596.1387&rep=rep1&type=pdf)

