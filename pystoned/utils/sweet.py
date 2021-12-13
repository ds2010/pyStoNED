# import dependencies
import numpy as np
from scipy.spatial.distance import cdist
from .tools import trans_list, to_2d_list


def sweet(x):
    """Sweet spot approach

    Args:
        x (float): input variables.

    Returns:
        list: active concavity constraint.
    """

    # transform data
    df = np.asmatrix(x)

    # calculate distance matrix
    distance = cdist(df, df)
    distance[np.diag_indices_from(distance)] = np.nan

    # calculate distance cut
    distcut = np.asmatrix(np.nanpercentile(distance, 3, axis=0))

    # find concavity constraint in sweet spot
    distance = np.where(np.isnan(distance), 0, distance)
    cutactive = np.zeros((len(distance), len(distance)))
    for i in range(len(distance)):
        for j in range(len(distance)):
            if distance[i, j] <= distcut[:, i]:
                cutactive[i, j] = 1

    return to_2d_list(trans_list(cutactive))
