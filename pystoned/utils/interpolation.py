# import dependencies
import numpy as np
from ..constant import FUN_PROD, FUN_COST


def interpolation(alpha, beta, x, fun=FUN_PROD):
    """Interpolate estimated function/frontier 

    Args:
        alpha (float): estimated alpha.
        beta (float): estimated beta.
        x (float): input variables.
        fun (String, optional): FUN_PROD (production frontier) or FUN_COST (cost frontier). Defaults to FUN_PROD.

    Returns:
        yat: interpolated frontier
    """
    n = len(x)
    d = len(x[0])

    if fun == FUN_PROD:
        def fun_est(x):
            return min(alpha + np.sum(beta[:, 0:d] * np.tile(x, (len(beta[:, 0:d]), 1)), axis=1))
    elif fun == FUN_COST:
        def fun_est(x):
            return max(alpha + np.sum(beta[:, 0:d] * np.tile(x, (len(beta[:, 0:d]), 1)), axis=1))

    yhat = np.zeros((n, 1))
    for i in range(n):
        yhat[i, :] = fun_est(x[i, :])

    return yhat
