import numpy as np


def dmdc(y, x):
    """distance matrix and distance cut"""

    # transform data
    df = np.asmatrix(x)

    # calculate distance matrix
    distmax = 0.0
    distmin = 1000000

    distance = np.zeros((len(df), len(df)))
    for i in range(len(df)):
        for j in range(len(df)):
            if i != j:
                distance[i, j] = np.sqrt(np.sum(np.square(df[i, :] - df[j, :])))
            if distance[i, j] > distmax:
                distmax = distance[i, j]
            if distance[i, j] < distmin:
                distmin = distance[i, j]

    # calculate distance cut
    distcut = np.zeros((1, len(df)))
    for i in range(len(distance)):
        distcut[:, i] = np.percentile(distance[:, i], 3)

    return distance, distcut
