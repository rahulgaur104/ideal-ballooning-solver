#!/usr/bin/env python3


import numpy as np
import pdb

from matplotlib import pyplot as plt

A = np.loadtxt("cobra_grate.NCSX_og")

ns1 = int(A[0, 2])

nangles = int(np.shape(A)[0]/(ns1+1))
B = np.zeros((ns1, ))
for i in range(nangles):
    if i == 0:
        B = A[i+1:(i+1)*ns1+1, 2]
    else:
        B = np.vstack((B, A[i*ns1+i+1:(i+1)*ns1+i+1, 2]))

gamma1 = np.amax(B, axis=0)

s1 = np.linspace(0, 1, ns1)
s1 = s1 + np.diff(s1)[0]

# gamma_max from my ideal-ballooning-solver
gamma2 = np.load("gamma_max_og.npy")
s2 = np.linspace(0.05, 0.995, len(gamma2))

# sincel the normalizations are different, we scale the gamma from our
#solver with the ratio of mean positive gamma values from both results
#gamma_scaling = np.mean(gamma1[gamma1>=0])/np.mean(gamma2[gamma2>0])
gamma_scaling = np.max(gamma1)/np.max(gamma2)

plt.plot(s1, gamma1, linewidth=2.5)
plt.plot(s2, gamma2*gamma_scaling, linewidth=2.5)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.xlabel(r"$s$", fontsize=16)
plt.ylabel(r"$\lambda$", fontsize=16)
plt.legend(["COBRAVMEC", "ballopt(scaled)"], fontsize=16)
plt.tight_layout()
plt.savefig("NCSX_unstable_comparison.png", dpi=600)
plt.show()
    







