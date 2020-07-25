"""
@title  : create binary matrix P (dominance relation)
@Author : Sheng Dai, Timo Kuosmanen
@Mail   : sheng.dai@aalto.fi (S. Dai); timo.kuosmanen@aalto.fi (T. Kuosmanen)
@Date   : 2020-04-21
"""

import numpy as np


def bimatp(x):

    # transform data
    x = np.array(x)

    # number of DMUs
    n = len(x)

    # number of inputs
    if type(x[0]) == int or type(x[0]) == float:
        m = 1
    else:
        m = len(x[0])

    # binary matrix P
    p = np.zeros((n, n))

    if m == 1:
        for i in range(n):
            p[i, :] = np.where(x[i, 0] <= x[:, 0], 1, 0)

    if m == 2:
        for i in range(n):
            p[i, :] = np.where((x[i, 0] <= x[:, 0]) & (x[i, 1] <= x[:, 1]), 1, 0)

    if m == 3:
        for i in range(n):
            p[i, :] = np.where((x[i, 0] <= x[:, 0]) & (x[i, 1] <= x[:, 1]) & (x[i, 2] <= x[:, 2]), 1, 0)

    if m == 4:
        for i in range(n):
            p[i, :] = np.where((x[i, 0] <= x[:, 0]) & (x[i, 1] <= x[:, 1]) & (x[i, 2] <= x[:, 2]) & (x[i, 3] <= x[:, 3]), 1, 0)

    if m == 5:
        for i in range(n):
            p[i, :] = np.where((x[i, 0] <= x[:, 0]) & (x[i, 1] <= x[:, 1]) & (x[i, 2] <= x[:, 2]) & (x[i, 3] <= x[:, 3]) & \
                               (x[i, 4] <= x[:, 4]), 1, 0)

    if m == 6:
        for i in range(n):
            p[i, :] = np.where((x[i, 0] <= x[:, 0]) & (x[i, 1] <= x[:, 1]) & (x[i, 2] <= x[:, 2]) & (x[i, 3] <= x[:, 3]) & \
                               (x[i, 4] <= x[:, 4]) & (x[i, 5] <= x[:, 5]), 1, 0)

    p = p.tolist()

    return p