import numpy as np
from scipy.optimize import linprog
from .tools import trans_list, to_1d_list, to_2d_list
from ..constant import RTS_VRS, RTS_CRS


def gmin(x, yhat, rts):

    if rts == RTS_VRS:
        
        x0 = np.ones(len(yhat))
        x = np.column_stack((x0, x))
        A = to_2d_list(trans_list(-x))
        b = to_1d_list(trans_list(-yhat))
        bounds = [(None, None)] + [(0, None) for _ in range(len(A[0]) - 1)]

        gmin = []
        for i in range(len(A)):
            c = x[i, :]
            res = linprog(c, A_ub=A, b_ub=b, bounds=bounds, method='highs')
            gmin.append(res.fun)

    elif rts == RTS_CRS:

        x = np.array(x)
        A = to_2d_list(trans_list(-x))
        b = to_1d_list(trans_list(-yhat))
        bounds = [(0, None) for _ in range(len(A[0]))]

        gmin = []
        for i in range(len(A)):
            c = x[i, :]
            res = linprog(c, A_ub=A, b_ub=b, bounds=bounds, method='highs')
            gmin.append(res.fun)

    return np.array(gmin)
