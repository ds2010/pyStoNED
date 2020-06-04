"""
@Title   : plot estimated CNLS frontier
@Author  : Sheng Dai, Timo Kuosmanen
@Mail    : sheng.dai@aalto.fi (S. Dai); timo.kuosmanen@aalto.fi (T. Kuosmanen)
@Date    : 2020-05-06
"""

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm


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


def cnlsplot3d(x, y, beta):
    
    # combine the array x, y, and f
    x  = np.array(x)
    y  = np.array(y)
    x  = x.reshape(-1,2).T
    x1 = x[0]
    x2 = x[1]
    f  = y - eps
    data = (np.stack([x1, x2, y, f], axis=0)).T

    # sort the array by first coloum (x1)
    col = 0
    data=data[np.argsort(data[:,col])].T
    
    x1 = data[0]
    x2 = data[1]
    y  = data[2]
    f  = data[3]
    f  = np.diag(f).T
  
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    
    # generate a mesh
    x_surf = x1              
    y_surf = x2
    x_surf, y_surf = np.meshgrid(x_surf, y_surf)
    z_surf = f            
    #ax.plot_surface(x_surf, y_surf, z_surf, cmap=cm.hot)    # plot a 3d surface plot
    ax.plot_surface(x_surf, y_surf, z_surf, rstride=1, cstride=1, cmap='rainbow')
       
    # plot 3d scatter
    ax.scatter(x1, x2, y)        

    ax.set_xlabel('x label')
    ax.set_ylabel('y label')
    ax.set_zlabel('z label')

    return fig













