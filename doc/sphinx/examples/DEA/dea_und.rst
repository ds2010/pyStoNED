=================================
DEA: Undesirable Output (DDF)
=================================


With help of DDF, we can resort to DEA to evulate the DMUs with undesirable output. The corresponding CRS and
VRS models are presented as follows

1. CRS

.. math::
    :nowrap:
    
    \begin{align*}
            \underset{\mathbf{\theta},\mathbf{\lambda }}max \quad \theta \\ 
            \mbox{s.t.} \quad 
            X\lambda \le x_o - \theta g_x   \\
            B\lambda = b_o - \theta g_b \\
            Y\lambda \ge y_o + \theta g_y\\
            \lambda \ge 0
    \end{align*}

2. VRS

.. math::
    :nowrap:
    
    \begin{align*}
            \underset{\mathbf{\theta},\mathbf{\lambda }}max \quad \theta \\ 
            \mbox{s.t.} \quad 
            X\lambda \le x_o - \theta g_x   \\
            B\lambda = b_o - \theta g_b \\
            Y\lambda \ge y_o + \theta g_y\\
            \sum_{j=1}^{n}\lambda_j = 1 \\
            \lambda \ge 0
    \end{align*}


Example: DEA-DDF with bad outputs `[.ipynb] <https://colab.research.google.com/github/ds2010/pyStoNED/blob/master/notebooks/DEA_UndesirableOutput.ipynb>`_
--------------------------------------------------------------------------------------------------------------------------------------


.. code:: python
    
    # import packages
    from pystoned import DEA
    from pystoned import dataset as dataset
    from pystoned.constant import RTS_VRS, OPT_LOCAL
        
    # import the GHG example data
    data = dataset.load_GHG_abatement_cost()
    
    # define and solve the DEA DDF model
    model = DEA.DDF(y=data.y, x=data.x, b=data.b, gy=[1], gx=[0.0, 0.0], gb=[-1], rts=RTS_VRS, yref=None, xref=None, bref=None)
    model.optimize(OPT_LOCAL)
    
    # display the technical efficiency
    model.display_theta()
    
    # display the intensity variables
    model.display_lamda()