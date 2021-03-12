============================
Additive Production function
============================

Hildreth (1954) was the first to consider nonparametric regression subject to 
monotonicity and concavity constraints in the case of a single input variable :math:`x`. 
Kuosmanen (2008) extended Hildreth’s approach to the multivariate setting with a 
vector-valued :math:`\bf{x}`, and coined the term convex nonparametric least squares (``CNLS``) for this method.
``CNLS`` builds upon the assumption that the true but unknown production function 
:math:`f` belongs to the set of continuous, monotonic increasing and globally concave functions, 
imposing exactly the same production axioms as standard DEA. 

The multivariate ``CNLS`` formulation is defined as:

.. math::
    :nowrap:
    
    \begin{align*}
      & \underset{\alpha, \beta, \varepsilon} {min} \sum_{i=1}^n\varepsilon_i^2 \\
      & \text{s.t.} \\
      &  y_i = \alpha_i + \beta_i^{'}X_i + \varepsilon_i \quad \forall i \\
      &  \alpha_i + \beta_i^{'}X_i \le \alpha_j + \beta_j^{'}X_i  \quad  \forall i, j\\
      &  \beta_i \ge 0 \quad  \forall i \\
    \end{align*}

where :math:`\alpha_i` and :math:`\beta_i` define the intercept and slope parameters of 
tangent hyperplanes that characterize the estimated piece-wise linear frontier. 
:math:`\varepsilon_i` denotes the CNLS residuals. The first constraint can be interpreted 
as a multivariate regression equation, the second constraint imposes convexity, 
and the third constraint imposes monotonicity.


Example `[.ipynb] <https://colab.research.google.com/github/ds2010/pyStoNED/blob/master/notebooks/CNLS_prod.ipynb>`_
------------------------------------------------------------------------------------------------------------------------------

In the following code, we estimate an additive production function with pyStoNED.

.. code:: python

    # import packages
    from pystoned import CNLS
    import pandas as pd
    import numpy as np
    
    # import Finnish electricity distribution firms data
    url='https://raw.githubusercontent.com/ds2010/pyStoNED/master/pystoned/data/electricityFirms.csv'
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
    model = CNLS.CNLS(y, x, z=None, cet = "addi", fun = "prod", rts = "vrs")
    model.optimize(remote=True)

    # display the estimates
    model.display_alpha()
    model.display_beta()
    model.display_residual()

    # store the estimates
    alpha = model.get_alpha()
    beta = model.get_beta()
    residuals = model.get_residual()


