=================================
DEA: Distance function
=================================


Chambers et al. (1996) introduced the directional distance function (DDF)
into efficiency measurement, and the inefficient DMUs can be projected to the
frontier using direction :math:`g = (−g_x , g_y) \neq 0_{m+s}`, where :math:`g_x \in R^m` and
:math:`g_y \in R^s`.

The VRA and CRS models are presented as follows

1. CRS
   
.. math::
    :nowrap:

    \begin{equation}
        \underset{\mathbf{\theta},\mathbf{\lambda }}max \quad \theta \\ 
        \mbox{s.t.} \quad 
        X\lambda \le x_o - \theta g_x   \\
        Y\lambda \ge y_o + \theta g_y\\
        \lambda \ge 0
    \end{equation}


2. VRS

.. math::
    :nowrap:

    \begin{equation}
        \underset{\mathbf{\theta},\mathbf{\lambda }}max \quad \theta \\ 
        \mbox{s.t.} \quad 
        X\lambda \le x_o - \theta g_x   \\
        Y\lambda \ge y_o + \theta g_y\\
        \sum_{j=1}^{n}\lambda_j = 1 \\
        \lambda \ge 0
    \end{equation}


Example: `[.ipynb] <https://colab.research.google.com/github/ds2010/pyStoNED/blob/master/notebooks/DEA_ddf_vrs.ipynb>`_
-----------------------------------------------------------------------------------------------------------------------------


.. code:: python
    
    # import packages
    from pystoned import DEA
    from pystoned import dataset as dataset
    from pystoned.constant import RTS_VRS, OPT_LOCAL
        
    # import the data provided with Tim Coelli’s Frontier 4.1
    data = dataset.load_Tim_Coelli_frontier()
        
    # define and solve the DEA DDF model
    model = DEA.DEADDF(y=data.y, x=data.x, b=None, gy=[1], gx=[0.0, 0.0], gb=None, rts=RTS_VRS, yref=None, xref=None, bref=None)
    model.optimize(OPT_LOCAL)
    
    # display the technical efficiency
    model.display_theta()
    
    # display the intensity variables
    model.display_lamda()