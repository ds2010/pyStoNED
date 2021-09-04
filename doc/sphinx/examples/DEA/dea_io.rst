=================================
Radial model: Input orientation
=================================

A set of :math:`j= 1,2,\cdots,n` observed ``DMUs`` transform a vector of :math:`i = 1, 2,\cdots,m`
inputs :math:`x \in R^m_{++}` into a vector of :math:`i = 1, 2, \cdots, s` outputs :math:`y \in R^s_{++}`
using the technology represented by the following **CRS** production possibility set: 
:math:`P_{crs} = \{(x, y) |x \ge X\lambda, y \le Y\lambda, \lambda \ge 0\}`, 
where :math:`X = (x)_j \in R^{s \times n}`, :math:`Y =(y)_j \in R^{m \times n}`
and :math:`\lambda = (\lambda_1, . . . , \lambda_n)^T` is a intensity vector. 

Based on the data matrix :math:`(X, Y)`, we measure the input oriented efficiency of 
``each observation o`` by solving ``n`` times the following linear programming problems: 

.. math::
    :nowrap:
    
    \begin{align*}
            \underset{\mathbf{\phi},\mathbf{\lambda }}min \quad \phi \\ 
            \mbox{s.t.} \quad 
            \phi x_o \ge X\lambda  \\
            Y\lambda \ge y_o \\
            \lambda \ge 0
    \end{align*}


The measurement of technical efficiency assuming **VRS** considers the following production 
possibility set :math:`P_{vrs} = \{ (x, y) |x \ge X\lambda, y \le Y\lambda, e\lambda = 1, \lambda \ge 0. \}`.
Thus, the only difference with the CRS model is the adjunction of the condition 
:math:`\sum_{j=1}^{n}\lambda_j = 1`. 

.. math::
    :nowrap:
    
    \begin{align*}
        \underset{\mathbf{\phi},\mathbf{\lambda }}min \quad \phi \\ 
        \mbox{s.t.} \quad 
        \phi x_o \ge X\lambda  \\
        Y\lambda \ge y_o \\
        \sum_{j=1}^{n}\lambda_j = 1 \\
        \lambda \ge 0
    \end{align*}


Example: Intput oriented DEA `[.ipynb] <https://colab.research.google.com/github/ds2010/pyStoNED/blob/master/notebooks/DEA_io_vrs.ipynb>`_
-----------------------------------------------------------------------------------------------------------------------------

In the following code, we calculate the VRS radial model with pyStoNED.

.. code:: python

    # import packages
    from pystoned import DEA
    from pystoned import dataset as dataset
    from pystoned.constant import RTS_VRS, ORIENT_IO, OPT_LOCAL
    
    # import the data provided with Tim Coelli’s Frontier 4.1
    data = dataset.load_Tim_Coelli_frontier()
    
    # define and solve the DEA radial model
    model = DEA.DEA(data.y, data.x, rts=RTS_VRS, orient=ORIENT_IO, yref=None, xref=None)
    model.optimize(OPT_LOCAL)

    # display the technical efficiency
    model.display_theta()

    # display the intensity variables
    model.display_lamda()
