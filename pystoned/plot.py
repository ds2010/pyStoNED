# import dependencies
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np
from .utils import interpolation
from .constant import FUN_PROD, FUN_COST, RED_MOM


def plot2d(model, x_select=0, label_name="estimated function", fig_name=None, method=RED_MOM):
    """Plot 2d estimated function/frontier

    Args:
        model: The input model for plotting.
        x_select (Integer): The selected x for plotting.
        label_name (String): the estimator name.
        fig_name (String, optional): The name of figure to save. Defaults to None.
    """
    x = np.array(model.x).T[x_select]
    y = np.array(model.y).T
    if y.ndim != 1:
        print("Plot with mutiple y is unavailable now.")
        return False
    if model.__class__.__name__ != "StoNED":
        yhat = np.array(model.get_frontier()).T
    else:
        yhat = np.array(model.get_frontier(method)).T

    data = (np.stack([x, y, yhat], axis=0)).T

    # sort
    data = data[np.argsort(data[:, 0])].T

    x, y, f = data[0], data[1], data[2]

    # create figure and axes objects
    fig, ax = plt.subplots()
    dp = ax.scatter(x, y, color="k", marker='x')
    fl = ax.plot(x, f, color="r", label=label_name)

    # add legend
    legend = plt.legend([dp, fl[0]],
                        ['Data points', label_name],
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
    if fig_name == None:
        plt.show()
    else:
        plt.savefig(fig_name)


def plot3d(model, x_select_1=0, x_select_2=1, fig_name=None, line_transparent=False, pane_transparent=False):
    """Plot 3d estimated function/frontier

    Args:
        model: The input model for plotting.
        x_select_1 (Integer): The selected x for plotting.
        x_select_2 (Integer): The selected x for plotting.
        fun (String, optional): FUN_PROD (production frontier) or FUN_COST (cost frontier). Defaults to FUN_PROD.       
        fig_name (String, optional): The name of figure to save. Defaults to None.
        line_transparent (bool, optional): control the transparency of the lines. Defaults to False.
        pane_transparent (bool, optional): control the transparency of the pane. Defaults to False.
    """
    x = np.array(model.x).T
    y = np.array(model.y).T
    alpha = model.get_alpha()
    beta = model.get_beta()
    if y.ndim != 1:
        print("Plot with mutiple y is unavailable now.")
        return False

    fig = plt.figure()
    ax = Axes3D(fig)
    dp = ax.scatter(x[x_select_1], x[x_select_2], y, marker='.', s=10)

    # Revise the Z-axis left side
    tmp_planes = ax.zaxis._PLANES
    ax.zaxis._PLANES = (tmp_planes[2], tmp_planes[3],
                        tmp_planes[0], tmp_planes[1],
                        tmp_planes[4], tmp_planes[5])

    # make the grid lines transparent
    if line_transparent == False:
        ax.xaxis._axinfo["grid"]['color'] = (1, 1, 1, 0)
        ax.yaxis._axinfo["grid"]['color'] = (1, 1, 1, 0)
        ax.zaxis._axinfo["grid"]['color'] = (1, 1, 1, 0)

    # make the panes transparent
    if pane_transparent != False:
        ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))

    xlin1 = np.linspace(min(x[x_select_1]), max(x[x_select_1]), 30)
    xlin2 = np.linspace(min(x[x_select_2]), max(x[x_select_2]), 30)
    XX, YY = np.meshgrid(xlin1, xlin2)

    ZZ = np.zeros((len(XX), len(XX)))
    for i in range(len(XX)):
        for j in range(len(XX)):
            ZZ[i, j] = interpolation.interpolation(alpha, beta,
                                                   x=np.array(
                                                       [XX[i, j], YY[i, j]], ndmin=2),
                                                   fun=model.fun)

    fl = ax.plot_surface(XX, YY, ZZ, rstride=1, cstride=1, cmap='viridis',
                         edgecolor='none', alpha=0.5)

    # add x, y, z label
    ax.set_xlabel("Input $x1$")
    ax.set_ylabel("Input $x2$")
    ax.set_zlabel("Output $y$", rotation=0)

    if fig_name == None:
        plt.show()
    else:
        plt.savefig(fig_name)
