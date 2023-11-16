We present a simple test to calculate and compare the marginal ideal ballooning stability boundary of a low-beta, large aspect ratio, shifted circle tokamak equilibrium.

Please ensure that your Python environment has the multiprocessing package. If not, use

```
pip install multiprocess
```

The expression for the coefficients g, c and, f and marginal stability diagram is given in Chris Bishop's notes [here](https://inis.iaea.org/search/search.aspx?orig_q=RN:17000660). 

The expression of the the geometric coefficients g, c and f is also given Connor, Hastie and Taylor's 1978 [PRL](https://journals.aps.org/prl/pdf/10.1103/PhysRevLett.40.396) on ideal ballooning mode.
However, since the actual marginal stability curve is not presented in Connor, Hastie and Taylor's paper, you should check out section 4 in Hudson and Hegna's [paper](https://doi.org/10.1063/1.1622669).

You may have to increase the resolution (len1 and len2) to get a smoother marginal stability diagram.






