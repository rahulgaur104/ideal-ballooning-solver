# ballopt: stellarator and tokamak stability optimizer

[![DOI](https://zenodo.org/badge/523432429.svg)](https://zenodo.org/badge/latestdoi/523432429)

This repository contains a set of Python scripts that, along with the SIMSOPT package can optimize or evaluate a tokamak or stellarator equilibrium against the infinite-n, ideal ballooning mode using an adjoint-based method.

The details of implementation of this method and the results are available in the paper [An adjoint-based method to optimize MHD equilibria against the infinite-n ideal ballooning mode](https://doi.org/10.1017/S0022377823000995)


## Dependencies

This optimizer is designed to run a cluster or supercomputer that uses SLURM. To run ballopt, you also need a Python version >= 3.8 and either

 * a [SIMSOPT singularity image\_v0.13](https://simsopt.readthedocs.io/en/latest/containers.html#singularity-container) to run this optimizer
 * or SIMSOPT Python environment and the VMEC2000 code from the [SIMSOPT GitHub page](https://github.com/hiddenSymmetries/simsopt) and make three one-line changes to the Simsopt\_submit.py (uncomment line 39) and ball\_submit.py (line 57) and slurm\_x0.sl (line 11) files.

## Instructions to reproduce the results in the adjoint-ballooning paper

The input files for the equilibria studied in the paper above are presented in this repository. You should be able to optimize any of the three equilibria with the following equilibrium names:

0. D3D
1. NCSX
2. HBERG

Depending on which equilibrium you want to optimize, run

python3  sims\_runner\_\<equilibrium\_name\>.py

from your login node. After each iteration, the data are saved in the directory save\_n\_load.

Please do not hesitate to contact me at rgaur@terpmail.umd.edu if you need any help running or understanding this optimizer.

## Update 2025: A new, GPU-accelerated version of this solver and optimizer is available as a part of the DESC code [here](https://github.com/PlasmaControl/DESC/blob/master/docs/notebooks/tutorials/ideal_ballooning_stability.ipynb).

## Relevant papers
* [An adjoint-based method to optimize MHD equilibria against the infinite-n ideal ballooning mode](https://doi.org/10.1017/S0022377823000995)
* [SIMSOPT: A flexible framework for stellarator optimization](https://joss.theoj.org/papers/10.21105/joss.03525)
* [COBRA: An Optimized Code for Fast Analysis of Ideal Ballooning Stability of Three-Dimensional Magnetic Equilibria](https://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.596.1387&rep=rep1&type=pdf)

