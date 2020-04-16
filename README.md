# StoNED-Python

`StoNED-Python` project provides the python codes for estimating Convex Nonparametric Least Square (`CNLS`), Stochastic Nonparametric Envelopment of Data (`StoNED`), and their different variants. It allows the user to estimate the CNLS/StoNED models in an open-access environment rather than in commercial software, e.g., GAMS, MATLAB. The `StoNED-Python` project is built based on the [PYOMO](http://www.pyomo.org/). 

# Authors
 + [Timo Kuosmanen](https://people.aalto.fi/timo.kuosmanen), Professor, Aalto University School of Business.
 + [Sheng Dai](https://www.researchgate.net/profile/Sheng_Dai8), Ph.D. candidate, Aalto University School of Business.

# To do list
- [ ] Additive model
   - [x] `CNLS`/`StoNED` (variables returns to scale (`VRS`) and constant returns to scale (`CRS`) models)
   - [x]  Residuals decomposition by method of moments(`MoM`) and quasi-likelihood estimation(`QLE`)
   - [ ]  Residuals decomposition by nonparametric kernel deconvolution (`NKD`)
- [ ] Multiplicative model
- [ ] Z-variables
- [ ] Convex quantile regression (`CQR`)
- [ ] Convex expectile regression (`CER`)
- [ ] Isotonic CNLS (`ICNLS`)
- [ ] Isotonic convex quantile regression (`ICQR`)
- [ ] Isotonic convex expectile regression (`ICER`)
