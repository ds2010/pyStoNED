===========================
FDH: Input orientation
===========================

The FDH estimator was first proposed by Deprins et al. (1984), and the mixed
integer linear program (MILP) formulation of FDH was introduced by Tulkens (1993).

Given the input variables :math:`X = [x_1, x_2, \cdots, x_n]` and output variables :math:`Y = [y_1, y_2, \cdots, y_n]`,
we measure the input oriented efficiency for observation :math:`i` by solving the following MILP problem: 

.. math::
    :nowrap:
    
    \begin{align*}
            \underset{\mathbf{\phi},\mathbf{\lambda}}min \quad \phi_i \\ 
            \mbox{s.t.} \quad
            X\lambda \le \phi_i x_i  \\
            Y\lambda \ge y_i  \\
            \sum \lambda = 1 \\
            \lambda_j \in \{0, 1\}, \forall j
    \end{align*}

where :math:`\lambda = [\lambda_1, \lambda_2, \cdots, \lambda_n]` is the vector of intensity weights. The efficiency
of observation :math:`i` is :math:`\phi^*_i`. The corresponding calculation processes are as follow: 


Example `[.ipynb] <https://colab.research.google.com/github/ds2010/pyStoNED/blob/master/notebooks/FDH_io.ipynb>`_
--------------------------------------------------------------------------------------------------------------------
    
        .. code:: python
        
            # import packages
            from pystoned import FDH
            from pystoned import dataset as dataset
            from pystoned.constant import ORIENT_IO, OPT_LOCAL
            
            # import the data provided with Tim Coelli’s Frontier 4.1
            data = dataset.load_Tim_Coelli_frontier()
            
            # define and solve the FDH model
            model = FDH.FDH(data.y, data.x, orient=ORIENT_IO)
            model.optimize(OPT_LOCAL)
        
            # display the technical efficiency
            model.display_theta()
        
            # display the intensity variables
            model.display_lamda()
