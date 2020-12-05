============================
Multiplicative cost function
============================

Similarly, we consider the follwing muliplicatiive cost models:

1. VRS model:

.. math::
    :nowrap:

    \begin{align*}
        & \underset{\alpha, \beta, \varepsilon} {min} \sum_{i=1}^n\varepsilon_i^2 \\
        & \text{s.t.} \\
        &  \text{ln}y_i = \text{ln}(\phi_i+1) + \varepsilon_i  \quad \forall i\\
        & \phi_i  = \alpha_i+\beta_i^{'}X_i -1 \quad \forall i \\
        &  \alpha_i + \beta_i^{'}X_i \ge \alpha_j + \beta_j^{'}X_i  \quad  \forall i, j\\
        &  \beta_i \ge 0 \quad  \forall i \\
    \end{align*}


2. CRS model:

.. math::
    :nowrap:
    
    \begin{align*}
        & \underset{\alpha, \beta, \varepsilon} {min} \sum_{i=1}^n\varepsilon_i^2 \\
        & \text{s.t.} \\
        &  \text{ln}y_i = \text{ln}(\phi_i+1) + \varepsilon_i  \quad \forall i\\
        &   \phi_i  = \beta_i^{'}X_i -1 \quad \forall i \\
        &  \beta_i^{'}X_i \ge \beta_j^{'}X_i  \quad  \forall i, j\\
        &  \beta_i \ge 0 \quad  \forall i \\
    \end{align*}

where :math:`\phi_i+1` is the CNLS estimator of :math:`E(y_i|x_i)`. The value of one is added here 
to make sure that the computational algorithms do not try to take logarithm of zero. 
The first equality can be interpreted as the log transformed regression equation 
(using the natural logarithm function :math:`ln(.)`). The rest of constraints 
are similar to additive production cost model. The use of :math:`\phi_i` allows
the estimation of a multiplicative relationship between output and 
input while assuring convexity of the production possibility set in original 
input-output space.


Example `[.ipynb] <https://colab.research.google.com/github/ds2010/pyStoNED/blob/master/notebooks/CNLS_mult_cost.ipynb>`_
-----------------------------------------------------------------------------------------------------------------------------------

In the following code, we estimate two multiplicative cost functions with pyStoNED.

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
    y = df['TOTEX']

    # inputs
    x1 = df['Energy']
    x1 = np.asmatrix(x1).T
    x2 = df['Length']
    x2 = np.asmatrix(x2).T
    x3 = df['Customers']
    x3 = np.asmatrix(x3).T
    x = np.concatenate((x1, x2, x3), axis=1)

    # define and solve the Multiplicative CNLS_vrs model
    model1 = CNLS.CNLS(y, x, z=None, cet = "mult", fun = "cost", rts = "vrs")
    model1.optimize(remote=True)

    # define and solve the Multiplicative CNLS_crs model
    model2 = CNLS.CNLS(y, x, z=None, cet = "mult", fun = "cost", rts = "crs")
    model2.optimize(remote=True)

    # print residuals in the VRS model
    print(model1.display_residual())

    # print residuals in the CRS model
    print(model2.display_residual())
