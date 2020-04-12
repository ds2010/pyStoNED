# -*- coding: utf-8 -*-
"""
@title : Convex Nonparametric Least Square (CNLS) example
@author: Sheng Dai (sheng.dai@aalto.fi)
@data  : Timo Kuosmanen (timo.kuosmanen@aalto.fi)
         Finnish Electricity Distribution Data       
@obj.  : Calculates a production function and overall economic efficiency         
"""
import CNLS
import pandas as pd
import numpy as np
from cvxopt import matrix

# import data
df = pd.read_excel ('Book1.xlsx')

# output
y  = df['Energy']

# inputs
x1 = df['OPEX']
x1 = np.asmatrix(x1).T
x2 = df['CAPEX']
x2 = np.asmatrix(x2).T

x  = np.concatenate((x1, x2), axis=1)

# call function cnls() and estimate the production function
res   = CNLS.cnls(x,y)

# results: yhat, g1, and g2
n     = len(y)  
yhat  = res[:n]
g1    = res[n:2*n]
g2    = res[2*n:]

eps   = matrix(0.0, (n,1))
alpha = matrix(0.0, (n,1))

# compute the residual (eps) and constant term (alpha)
for i in range(n):
    eps[i]   = y[i] - yhat[i]
    alpha[i] = y[i] - eps[i] - g1[i] * x1[i] - g2[i] * x2[i]    
    
alpha = pd.DataFrame(alpha)
yhat  = pd.DataFrame(yhat)
g1    = pd.DataFrame(g1)
g2    = pd.DataFrame(g2)
eps   = pd.DataFrame(eps)

results = pd.concat([g1, g2, alpha, eps], axis=1) 

# export the estimates
results.to_excel("CNLS_prod.xlsx", sheet_name='CNLS_prod')