"""
@title  : quasi-Likelihood function for residual (eps) given normal-half normal
          distribution and parameter lambda
@Author : Sheng Dai, Timo Kuosmanen
@Mail   : sheng.dai@aalto.fi (S. Dai); timo.kuosmanen@aalto.fi (T. Kuosmanen)  
@Date   : 2020-04-12 
"""

import numpy as np
import math
from scipy.stats import norm

# production frontier
def qlep(lamda, eps):

    # sigma Eq. (3.26) in Johnson and Kuosmanen (2015)
    sigma = np.sqrt(np.mean(eps ** 2) / (1 - 2 * lamda ** 2 / (math.pi * (1 + lamda ** 2))))

    # bias adjusted residuals Eq. (3.25)
    # mean
    mu = math.sqrt(2 / math.pi) * sigma * lamda / math.sqrt(1 + lamda ** 2)

    # adj. res.
    epsilon = eps - mu

    # log-likelihood function Eq. (3.24)
    pn = norm.cdf(-epsilon * lamda / sigma)
    logl = -len(eps) * math.log(sigma) + np.sum(np.log(pn)) - 0.5 * np.sum(epsilon ** 2) / sigma ** 2

    return -logl

# cost frontier
def qlec(lamda, eps):

    # sigma Eq. (3.26) in Johnson and Kuosmanen (2015)
    sigma = np.sqrt(np.mean(eps ** 2) / (1 - 2 * lamda ** 2 / (math.pi * (1 + lamda ** 2))))

    # bias adjusted residuals Eq. (3.25)
    # mean
    mu = math.sqrt(2 / math.pi) * sigma * lamda / math.sqrt(1 + lamda ** 2)

    # adj. res.
    epsilon = eps + mu

    # log-likelihood function Eq. (3.24)
    pn = norm.cdf(epsilon * lamda / sigma)
    logl = -len(eps) * math.log(sigma) + np.sum(np.log(pn)) - 0.5 * np.sum(epsilon ** 2) / sigma ** 2

    return -logl