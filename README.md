# StoNED-Python

`StoNED-Python` project provides the python codes for estimating Convex Nonparametric Least Square (`CNLS`), Stochastic Nonparametric Envelopment of Data (`StoNED`), and other different `StoNED`-related variants. It allows the user to estimate the CNLS/StoNED models in an open-access environment rather than in commercial software, e.g., GAMS, MATLAB. The `StoNED-Python` project is built based on the [PYOMO](http://www.pyomo.org/). 

# Installation

We have published a beta version [`pyStoNED`](https://pypi.org/project/pystoned/) package on PyPI. Please feel free to download and test it. We welcome any bug reports and feedback.

#### PyPI [![PyPI version](https://img.shields.io/pypi/v/pystoned.svg?maxAge=3600)](https://pypi.org/project/pystoned/) [![PyPI downloads](https://img.shields.io/pypi/dm/pystoned.svg?maxAge=21600)](https://pypistats.org/packages/pystoned)

    pip install pystoned

# [Tutorials](https://github.com/ds2010/StoNED-Python/tree/master/Tutorials)

A number of `StoEND`-related tutorial examples are provided as follows, and more detailed technical reports are currently under development.
  + Convex Nonparametric Least Square (`CNLS`)
    + [Production function estimation](https://github.com/ds2010/StoNED-Python/blob/master/Tutorials/CNLS_prod.ipynb) 
    + [Cost function estimation](https://github.com/ds2010/StoNED-Python/blob/master/Tutorials/CNLS_cost.ipynb)    
  + Stochastic Nonparametric Envelopment of Data (`StoNED`)
    + [Log-transformed cost function estimation and residual decomposition by MoM](https://github.com/ds2010/StoNED-Python/blob/master/Tutorials/StoNED.ipynb)
    + [StoNED with Z-variable](https://github.com/ds2010/StoNED-Python/blob/master/Tutorials/StoNEZD.ipynb)
    + Estimating unconditional expected inefficiency ![equation](https://latex.codecogs.com/gif.latex?%5Cmu) by KDE
  + [Convex quantile/expectile regression (`CQR` and `CER`)](https://github.com/ds2010/StoNED-Python/blob/master/Tutorials/CQR_CER.ipynb)
  + Corrected convex nonparametric least squares (C<sup>2</sup>NLS)
  + Isotonic Convex Nonparametric Least Square (`ICNLS`)
  + Basic Data Envelopment Analysis (DEA) models
    + [Input oriented VRS model](https://github.com/ds2010/Productivity-and-Efficiency-Analysis/blob/master/DEA/DEA_VRS.ipynb) 
    + Output oriented VRS model

Additional tutorials:
  + [Rewrite GAMS codes in Python](https://github.com/ds2010/StoNED-Python/blob/master/Tutorials/gams2python.ipynb)
  + [A ConcreteModel for CNLS estimation](https://github.com/ds2010/StoNED-Python/blob/master/Tutorials/ConcreteModel.ipynb)
  + [Tutorials for Productivity and Efficiency Analysis](https://github.com/ds2010/Productivity-and-Efficiency-Analysis)
 
> **Note**
  1. The entire [function list](https://github.com/ds2010/StoNED-Python/blob/master/Tutorials/function_list.ipynb) and syntax in [`pyStoNED`](https://pypi.org/project/pystoned/) package can be seen from [Tutorials](https://github.com/ds2010/StoNED-Python/tree/master/Tutorials).
  2. The [list of acronyms](https://github.com/ds2010/StoNED-Python/blob/master/Tutorials/list_of_acronyms.ipynb) and symbols is available at [Tutorials](https://github.com/ds2010/StoNED-Python/tree/master/Tutorials).
  
# Authors

 + [Timo Kuosmanen](https://people.aalto.fi/timo.kuosmanen), Professor, Aalto University School of Business.
 + [Sheng Dai](https://www.researchgate.net/profile/Sheng_Dai8), Ph.D. candidate, Aalto University School of Business.

# To do list
- [x]  `CNLS`/`StoNED`
   - [x] Production function estimation
   - [x] Cost function estimation
   - [x] variables returns to scale (`VRS`) model
   - [x] constant returns to scale (`CRS`) model
   - [x] Additive composite error term
   - [x] Multiplicative composite error term
   - [x] Residuals decomposition by method of moments(`MoM`) 
   - [x] Residuals decomposition by quasi-likelihood estimation(`QLE`)
   - [x] Residuals decomposition by nonparametric kernel deconvolution (`NKD`)
- [x] `StoNEZD` (contextual variables)
- [x] Convex quantile regression (`CQR`)
- [x] Convex expectile regression (`CER`)
- [x] Isotonic CNLS (`ICNLS`)
- [ ] Isotonic convex quantile regression (`ICQR`)
- [ ] Isotonic convex expectile regression (`ICER`)
- [x] Corrected convex nonparametric least squares (C<sup>2</sup>NLS)
- [x] Multiple outputs (CNLS-DDF formulation)
   - [x] with undesirable outputs
   - [x] without undesirable outputs
- [x] Multiple outputs (CQR/CER-DDF formulation)
   - [x] with undesirable outputs
   - [x] without undesirable outputs   
- [ ] Representation of `StoNED`-related frontier/quantile function
   - [x] one input and one output
   - [ ] two inputs and one output 
   - [ ] three inputs and one output 
