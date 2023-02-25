# import dependencies
import numpy as np
from ..constant import FUN_PROD, FUN_COST
from .tools import trans_list, to_2d_list


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
    x = np.array(to_2d_list(trans_list(x)))
    n = len(x)
    d = len(x[0])

    if len(beta[0]) != d:
        raise ValueError(
            "The dimensions of x_test must be equal to those of x.")

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
