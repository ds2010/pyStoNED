# StoNED-Python

`StoNED-Python` project provides the python codes for estimating Convex Nonparametric Least Square (`CNLS`), Stochastic Nonparametric Envelopment of Data (`StoNED`), and other different `StoNED`-related variants. It allows the user to estimate the CNLS/StoNED models in an open-access environment rather than in commercial software, e.g., GAMS, MATLAB. The `StoNED-Python` project is built based on the [PYOMO](http://www.pyomo.org/). 

# Installation

We have published a beta version [`pyStoNED`](https://pypi.org/project/pystoned/) package on PyPI. Please feel free to download and test it. We welcome any bug reports and feedback.

#### PyPI [![PyPI version](https://img.shields.io/pypi/v/pystoned.svg?maxAge=3600)](https://pypi.org/project/pystoned/) [![Downloads](https://pepy.tech/badge/pystoned/month)](https://pepy.tech/project/pystoned/month) [![Downloads](https://pepy.tech/badge/pystoned)](https://pepy.tech/project/pystoned)

    pip install pystoned

#### GitHub -- Latest development version

    pip install -U git+https://github.com/ds2010/StoNED-Python

# [Tutorials](https://github.com/ds2010/pyStoNED-Tutorials)

A number of Jupyter Notebooks are provided in the repository [pyStoNED-Tutorials](https://github.com/ds2010/pyStoNED-Tutorials), and more detailed technical reports are currently under development.
  
# Authors

 + [Timo Kuosmanen](https://people.aalto.fi/timo.kuosmanen), Professor, Aalto University School of Business.
 + [Sheng Dai](https://www.researchgate.net/profile/Sheng_Dai8), Ph.D. candidate, Aalto University School of Business.
 + [Chia-Yen Lee](https://scholar.google.com/citations?user=M_DB0CQAAAAJ&hl=en), Professor, College of Management, National Taiwan University.
 + [Yu-Hsueh Fang](https://github.com/JulianATA), Computer Engineer, Institute of Manufacturing Information and Systems, National Cheng Kung University.

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
- [x] A more efficient algorithm for CNLS (`CNLSG`)  
- [x] `StoNEZD` (contextual variables)
- [x] Convex quantile regression (`CQR`)
- [x] Convex expectile regression (`CER`)
- [x] Isotonic CNLS (`ICNLS`)
- [x] Isotonic convex quantile regression (`ICQR`)
- [x] Isotonic convex expectile regression (`ICER`)
- [x] Corrected convex nonparametric least squares (C<sup>2</sup>NLS)
- [x] Multiple outputs (CNLS-DDF formulation)
   - [x] with undesirable outputs
   - [x] without undesirable outputs
- [x] Multiple outputs (CQR/CER-DDF formulation)
   - [x] with undesirable outputs
   - [x] without undesirable outputs   
- [x] Data Envelopment Analysis (`DEA`)
   - [x] Radial input oriented model: CRS and VRS
   - [x] Radial output oriented model: CRS and VRS
   - [x] Directional model: CRS and VRS
   - [x] Directional model with undesirable outputs: CRS and VRS
- [x] Free Disposal Hull (`FDH`) Analysis 
   - [x] Radial input oriented FDH model
   - [x] Radial output oriented FDH model            
- [ ] Representation of `StoNED`-related frontier/quantile function
   - [x] one input and one output
   - [ ] two inputs and one output 
   - [ ] three inputs and one output 
