# import dependencies
import matplotlib.pyplot as plt
import numpy as np


class plot2d:
    """2D estimated function/frontier
    """

    def __init__(self, x, y, f, label_name, fig_name=None):
        """Plot 2D estimated function/frontier

        Args:
            x (float): output variable. 
            y (float): input variables.
            f (float): estimated function?frontier
            label_name (String): the estimator name.
            fig_name (String, optional): The name of figure to save. Defaults to None.
        """
        x = np.array(self.x).T
        y = np.array(self.y).T
        f = np.array(f).T
        data = (np.stack([x, y, f], axis=0)).T

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


