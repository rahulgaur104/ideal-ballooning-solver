# ballopt: stellarator and tokamak stability optimizer

This repository contains a set of Python scripts that, along with the SIMSOPT package can optimize or evaluate a tokamak or stellarator equilibrium against the infinite-n, ideal ballooning mode using an adjoint-based method.

The details of implementation of this method and the results are available on arxiv: [An adjoint-based method to optimize MHD equilibria against the infinite-n ideal ballooning mode](https://arxiv.org/abs/2302.07673)

## Dependencies

If you are using a cluster or supercomputer that uses SLURM, you need a Python version >= 3.8 and a [SIMSOPT singularity image\_v0.13](https://simsopt.readthedocs.io/en/latest/containers.html#singularity-container) to run this optimizer. However, if you are running this on a laptop or a device without SLURM, you need to install the SIMSOPT optimization package and the VMEC2000 code from the [SIMSOPT GitHub page](https://github.com/hiddenSymmetries/simsopt) and make three one-line changes to the Simsopt\_runner.py (line 39) and ball\_scan.py (line 56) and slurm\_x0.sl (line 11) files.

## Instructions to reproduce the results in the adjoint-ballooning paper

The input files for the equilibria studied in the paper above are presented in this repository. You should be able to optimize any of the three equilibria with the following equilibrium names:

0. D3D
1. NCSX
2. HBERG

Depending on which equilibrium you want to optimize, run

python3  sims\_runner\_\<equilibrium\_name\>.py

from your login node. After each iteration, the data are saved in the directory save\_n\_load.

Please do not hesitate to contact me at rgaur@terpmail.umd.edu if you need any help running or understanding this optimizer.

## Relevant papers
* [An adjoint-based method to optimize MHD equilibria against the infinite-n ideal ballooning mode](https://arxiv.org/abs/2302.07673)
* [SIMSOPT: A flexible framework for stellarator optimization](https://joss.theoj.org/papers/10.21105/joss.03525)
* [COBRA: An Optimized Code for Fast Analysis of Ideal Ballooning Stability of Three-Dimensional Magnetic Equilibria](https://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.596.1387&rep=rep1&type=pdf)

