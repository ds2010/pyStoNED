# StoNED-Python

`StoNED-Python` project provides the python codes for estimating Convex Nonparametric Least Square (`CNLS`), Stochastic Nonparametric Envelopment of Data (`StoNED`), and their different variants. It allows the user to estimate the CNLS/StoNED models in an open access environment rather than in commercial softwares, e.g., GAMS, MATLAB. The `StoNED-Python` project is built based on the [CVXOPT](https://cvxopt.org/). 

It allows the user to formulate convex optimization problems in a natural mathematical syntax rather than the restrictive standard form required by most solvers. The user specifies an objective and set of constraints by combining constants, variables, and parameters using a library of functions with known mathematical properties. CVXR then applies signed disciplined convex programming (DCP) to verify the problem’s convexity. Once verified, the problem is converted into standard conic form using graph implementations and passed to a cone solver such as ECOS or SCS.


# Authors
 + [Timo Kuosmanen](https://people.aalto.fi/timo.kuosmanen), Professor, Aalto University School of Business.
 + [Sheng Dai](https://www.researchgate.net/profile/Sheng_Dai8), Ph.D. candidate, Aalto University School of Business.


# License
This project is licensed under the MIT License - see the [LICENSE.md](https://github.com/ds2010/StoNED-Python/blob/master/LICENSE) file for details

