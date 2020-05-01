"""
@title  : nonparametric kernel deconvolution for residual (eps) decomposition
@Author : Sheng Dai, Timo Kuosmanen
@Mail   : sheng.dai@aalto.fi (S. Dai); timo.kuosmanen@aalto.fi (T. Kuosmanen)
@Date   : 2020-05-01
"""

from sklearn.neighbors import KernelDensity
import numpy as np
from scipy import stats

def kd(eps, func):
    # func    = "prod" : production frontier
    #         = "cost" : cost frontier

    # choose a bandwidth (rule-of-thumb, Eq. (3.29) in Silverman (1986))
    std = np.std(eps, ddof=1)
    iqr = stats.iqr(eps, interpolation='midpoint')

    if std < iqr:
        sigmahat = std
    else:
        sigmahat = iqr / 1.349
    bw = 1.06 * sigmahat * len(eps) ** (-1 / 5)

    # reshape the array eps (due to only one feature/column)
    eps = eps.reshape(-1, 1)

    # fit the KDE model
    kde = KernelDensity(bandwidth=bw, kernel='gaussian')
    kde.fit(eps)

    # score_samples returns the log of the probability density
    logprob = kde.score_samples(eps)
    den = np.exp(logprob)

    # first derivative of density function
    epsD = np.zeros((len(eps), 1))
    denD = np.zeros((len(eps), 1))
    der = np.zeros((len(eps), 1))
    for i in range(len(eps) - 1):
        epsD[i + 1] = eps[i + 1] - eps[i]
        denD[i + 1] = den[i + 1] - den[i]
        der[i + 1] = 0.2 * denD[i + 1] / epsD[i + 1]

    # expected inefficiency mu
    if func == "prod":
        mu = -np.max(der)

    if func == "cost":
        mu = np.max(der)

    return mu