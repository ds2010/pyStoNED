========================
Additive cost function
========================

We now consider the ``CNLS`` that builds upon the assumption that the true but unknown production function 
:math:`f` belongs to the set of continuous, monotonic increasing and globally convex functions, 
imposing exactly the same production axioms as standard DEA. 

The multivariate ``CNLS`` formulation is defined as:

.. math::
    :nowrap:
    
    \begin{align*}
        & \underset{\alpha, \beta, \varepsilon} {min} \sum_{i=1}^n\varepsilon_i^2 \\
        & \text{s.t.} \\
        &  y_i = \alpha_i + \beta_i^{'}X_i + \varepsilon_i \quad \forall i \\
        &  \alpha_i + \beta_i^{'}X_i \ge \alpha_j + \beta_j^{'}X_i  \quad  \forall i, j\\
        &  \beta_i \ge 0 \quad  \forall i \\
    \end{align*}

where :math:`\alpha_i` and :math:`\beta_i` define the intercept and slope parameters of 
tangent hyperplanes that characterize the estimated piece-wise linear frontier. 
:math:`\varepsilon_i` denotes the CNLS residuals. The first constraint can be interpreted 
as a multivariate regression equation, the second constraint imposes concavity, 
and the third constraint imposes monotonicity.


Example `[.ipynb] <https://colab.research.google.com/github/ds2010/pyStoNED/blob/master/sources/notebooks/CNLS_cost.ipynb>`_
-----------------------------------------------------------------------------------------------------------------------------

In the following code, we estimate an additive cost function with pyStoNED.

.. code:: python

    # import packages
    from pystoned import CNLS
    import pandas as pd
    import numpy as np
    
    # import Finnish electricity distribution firms data
    url='https://raw.githubusercontent.com/ds2010/pyStoNED/master/sources/data/firms.csv'
    df = pd.read_csv(url, error_bad_lines=False)
    df.head(5)
    
    # output
    y = df['Energy']

    # inputs
    x1 = df['OPEX']
    x1 = np.asmatrix(x1).T
    x2 = df['CAPEX']
    x2 = np.asmatrix(x2).T
    x = np.concatenate((x1, x2), axis=1)

    # define and solve the CNLS model
    model = CNLS.CNLS(y, x, z=None, cet = "addi", fun = "cost", rts = "vrs")
    model.optimize(remote=True)

    # print the residuals
    print(model.get_residual())
