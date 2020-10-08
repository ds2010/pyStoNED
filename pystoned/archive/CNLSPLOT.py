"""
@Title   : plot estimated CNLS frontier
@Author  : Sheng Dai, Timo Kuosmanen
@Mail    : sheng.dai@aalto.fi (S. Dai); timo.kuosmanen@aalto.fi (T. Kuosmanen)
@Date    : 2020-05-06
"""

import matplotlib.pyplot as plt
import numpy as np


def cnlsplot2d(x, y, eps):

    # combine the array x, y, and f
    x = np.array(x)
    y = np.array(y)
    x = x.reshape(-1)
    f = y - eps
    data = (np.stack([x, y, f], axis=0)).T

    # sort the array by first column (x)
    col = 0
    data=data[np.argsort(data[:,col])].T
    
    x = data[0]
    y = data[1]
    f = data[2]
    
    # create figure and axes objects
    fig, ax = plt.subplots()
    dp = ax.scatter(x, y, color="k", marker='x')
    fl = ax.plot(x, f, color="r", label="CNLS")
     
    # add legend
    legend = plt.legend([dp, fl[0]], 
               ['Data points', 'CNLS'],
               loc='upper left',
               ncol=1,
               fontsize=10,
               frameon=False)
    
    # add x, y label 
    ax.set_xlabel("Input $x$")
    ax.set_ylabel("Output $y$")
    
    # Remove top and right axes
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)

    # adjust the axis 
    #plt.axis([0, max(x)+1, 0, max(y)+1]) 

    plt.margins(x=0)
    
    return fig













