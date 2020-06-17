"""
@Title   : Corrected Convex Nonparametric Least Squares (C2NLS)
@Author  : Sheng Dai, Timo Kuosmanen
@Mail    : sheng.dai@aalto.fi (S. Dai); timo.kuosmanen@aalto.fi (T. Kuosmanen)  
@Date    : 2020-04-16
"""

import numpy as np


def ccnls(eps, alpha):

    # CCNLS efficiency estimator
    eps_ccnls = eps - np.amax(eps)

    # adjusted CCNLS's intercept alpha
    alpha_ccnls = alpha + np.amax(eps)

    return eps_ccnls, alpha_ccnls
