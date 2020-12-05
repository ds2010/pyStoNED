==============
Corrected CNLS 
==============

Corrected  nonparametric  least  squares (``C2NLS``) is  a  new  nonparametric  variant  
of the COLS model in which nonparametric least squares subject to monotonicity and 
concavity constraints replace thefirst-stage parametric OLS regression. The ``C2NLS``
model assumes that the regression :math:`f` is monotonic increasing and globally concave, the 
inefficiencies :math:`\varepsilon` are identically and independently distributed with 
mean :math:`\mu` and a finitevariance :math:`\sigma^2`, and that the inefficiencies :math:`\varepsilon` are 
uncorrelatedwith inputs :math:`\bf X`.

Like ``COLS``, the ``C2NLS`` method is implemented in two stages, which can be stated as follows:

* First stage: Estimate :math:`E(y_i|x_i)` by solving the following CNLS problem. Denote the CNLS residuals by :math:`\varepsilon^{CNLS}_i`.


.. math::
    :nowrap:

    \begin{align*}
        & \underset{\alpha, \beta, \varepsilon} {min} \sum_{i=1}^n\varepsilon_i^2 \\
        & \text{s.t.} \\
        &  y_i = \alpha_i + \beta_i^{'}X_i + \varepsilon_i \quad \forall i \\
        &  \alpha_i + \beta_i^{'}X_i \le \alpha_j + \beta_j^{'}X_i  \quad  \forall i, j\\
        &  \beta_i \ge 0 \quad  \forall i \\
    \end{align*}

* Second stage: Shift the residuals analogous to the ``COLS`` procedure; the ``C2NLS`` efficiency estimator is


.. math::
    :nowrap:

    \begin{align*}
        \hat{\varepsilon_i}^{C2NLS}= \varepsilon_i^{CNLS}− \max_h \varepsilon_h^{CNLS},
    \end{align*}

where values of :math:`\hat{\varepsilon_i}^{C2NLS}` range from :math:`[0, +\infty]` with 0 
indicating efficient performance. Similarly, we adjust the CNLS intercepts :math:`\alpha_i` as


.. math::
    :nowrap:
    
    \begin{align*}
        \hat{\alpha_i}^{C2NLS}= \alpha_i^{CNLS} + \max_h \varepsilon_h^{CNLS},
    \end{align*}


where :math:`\alpha_i^{CNLS}` is the optimal intercept for firmi in above CNLS problem
and :math:`\alpha_i^{C2NLS}` is the ``C2NLS`` estimator. Slope coefficients :math:`\beta_i` 
for ``C2NLS`` are obtained directly as the optimal solution to the CNLS problem.


Example `[.ipynb] <https://colab.research.google.com/github/ds2010/pyStoNED/blob/master/notebooks/CCNLS.ipynb>`_
------------------------------------------------------------------------------------------------------------------------------

In the following code, we estimate an Corrected CNLS model by using the pyStoNED.

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

    # First stage
    # define and solve the CNLS model
    model = CNLS.CNLS(y, x, z= None, cet = "addi", fun = "prod", rts = "vrs")
    model.optimize(remote=True)

    # Second stage
    # print the shifted residuals
    print(model.get_adjusted_residual())    
    # print the shifted intercept
    print(model.get_adjusted_alpha())   
