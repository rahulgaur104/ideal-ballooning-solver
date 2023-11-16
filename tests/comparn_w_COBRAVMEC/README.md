We also compare the growth rates with another ideal ballooning solver COBRAVMEC. The COBRAVMEC input (cobra.input\_\*) and output files (cobra\_grate.\*) and the ballopt \*.npy files are given in this repo along with the Python script used to read and plot the growth rates. 

Since COBRAVMEC uses a different normalization than ballopt, we had to scale the ballopt results with the ratio of the maximum growth rates for the unstable equilibria and minimum growth rates for the stable equilibria. 

We see that the growth rates do not agree that well (possibly due to resolution and operational differences b/w the solvers). However, the optimizer works reasonably well because the points of marginal stability are close enough. A stable ballopt equilibrium is also a stable COBRAVMEC equilibrium. 

A few points of comparison:

* In general, ballopt provides the user with a smoother gamma\_max than COBRAVMEC (for the same runtime) because we use a adjoint-based local optimizer on each flux surface to find gamma\_max. COBRAVMEC requires the user to provide the points on a surface at which they want the growth rate. As you can see, for the hberg case, it is hard to find gamma max just by trial and error.
* This smoothness of gamma is advantageous for performing gradient based optimization.



