# StoNED-Python

`StoNED-Python` project provides the python codes for estimating Convex Nonparametric Least Square (`CNLS`), Stochastic Nonparametric Envelopment of Data (`StoNED`), and other different `StoNED`-related variants. It allows the user to estimate the CNLS/StoNED models in an open-access environment rather than in commercial software, e.g., GAMS, MATLAB. The `StoNED-Python` project is built based on the [PYOMO](http://www.pyomo.org/). 

# Installation

We have published a beta version [`pyStoNED`](https://pypi.org/project/pystoned/) package on PyPI. Please feel free to download and test it. We welcome any bug reports and feedback.

#### PyPI [![PyPI version](https://img.shields.io/pypi/v/pystoned.svg?maxAge=3600)](https://pypi.org/project/pystoned/) [![Downloads](https://pepy.tech/badge/pystoned/month)](https://pepy.tech/project/pystoned/month) [![Downloads](https://pepy.tech/badge/pystoned)](https://pepy.tech/project/pystoned)

    pip install pystoned

# [Tutorials](https://github.com/ds2010/StoNED-Python/tree/master/Tutorials)

A number of Jupyter Notebooks are provided in the repository [pyStoNED-Tutorials](https://github.com/ds2010/pyStoNED-Tutorials), and more detailed technical reports are currently under development.
  
# Authors

 + [Timo Kuosmanen](https://people.aalto.fi/timo.kuosmanen), Professor, Aalto University School of Business.
 + [Sheng Dai](https://www.researchgate.net/profile/Sheng_Dai8), Ph.D. candidate, Aalto University School of Business.

# Available models
- `CNLS`/`StoNED`
  - Production function estimation
  - Cost function estimation
  - variables returns to scale (`VRS`) model
  - constant returns to scale (`CRS`) model
  - Additive composite error term
  - Multiplicative composite error term
  - Residuals decomposition by method of moments(`MoM`) 
  - Residuals decomposition by quasi-likelihood estimation(`QLE`)
  - Residuals decomposition by nonparametric kernel deconvolution (`NKD`)
- `StoNEZD` (contextual variables)
- Convex quantile regression (`CQR`)
- Convex expectile regression (`CER`)
- Isotonic CNLS (`ICNLS`)
- Isotonic convex quantile regression (`ICQR`)
- Isotonic convex expectile regression (`ICER`)
- Corrected convex nonparametric least squares (C<sup>2</sup>NLS)
- Multiple outputs (CNLS-DDF formulation)
  - with undesirable outputs
  - without undesirable outputs
- Multiple outputs (CQR/CER-DDF formulation)
  - with undesirable outputs
  - without undesirable outputs   
- Basic Data Envelopment Analysis (`DEA`) models
  - Radial input oriented model: CRS and VRS
  - Radial output oriented model: CRS and VRS
  - Directional model: CRS and VRS
  - Directional model with undesirable outputs: CRS and VRS         
- Representation of `StoNED`-related frontier/quantile function
  - one input and one output 
