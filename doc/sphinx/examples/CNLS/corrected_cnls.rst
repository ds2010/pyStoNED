========================
Corrected CNLS model
========================

Corrected convex nonparametric least squares (:math:`\text{C}^2\text{NLS}`) is a variant of the corrected ordinary least squares (COLS) 
model in which nonparametric least squares subject to monotonicity and concavity constraints replace the first-stage 
parametric ordinary least squares (OLS) regression. To estimate the production models, the :math:`\text{C}^2\text{NLS}` model assumes 
that the regression :math:`f` is monotonic increasing and globally concave production function, the inefficiencies 
:math:`\varepsilon` are identically and independently distributed with mean :math:`\mu` and a finite variance :math:`\sigma^2`, 
and that the inefficiencies :math:`\varepsilon` are uncorrelated with inputs :math:`\boldsymbol{x}`.

Like COLS, the :math:`\text{C}^2\text{NLS}` method is implemented in two stages, which can be stated as follows:

* Estimate :math:`E(y_i|x_i)` by solving the CNLS model, e.g., Problem (2.2). Denote the CNLS residuals by :math:`\varepsilon^{CNLS}_i`.

* Shift the residuals analogous to the ``COLS`` procedure; the ``C2NLS`` efficiency estimator is

.. math::
    :nowrap:

    \begin{align*}
        \hat{\varepsilon_i}^{C2NLS}= \varepsilon_i^{CNLS} − \max_h \varepsilon_h^{CNLS},
    \end{align*}

where values of :math:`\hat{\varepsilon_i}^{C2NLS}` range from :math:`[0, +\infty]` with 0 
indicating efficient performance. Similarly, we adjust the CNLS intercepts :math:`\alpha_i` as


.. math::
    :nowrap:
    
    \begin{align*}
        \hat{\alpha_i}^{C2NLS}= \alpha_i^{CNLS} + \max_h \varepsilon_h^{CNLS},
    \end{align*}


where :math:`\alpha_i^{CNLS}` is the optimal intercept for firmi in above CNLS problem
and :math:`\alpha_i^{C2NLS}` is the :math:`\text{C}^2\text{NLS}` estimator. Slope coefficients :math:`\beta_i` 
for :math:`\text{C}^2\text{NLS}` are obtained directly as the optimal solution to the CNLS problem.


Example: Corrected CNLS `[.ipynb] <https://colab.research.google.com/github/ds2010/pyStoNED/blob/master/notebooks/CCNLS.ipynb>`_
---------------------------------------------------------------------------------------------------------------------------------

.. code:: python

    # import packages
    from pystoned import CNLS
    from pystoned.constant import CET_ADDI, FUN_PROD, OPT_LOCAL, RTS_VRS
    from pystoned.dataset import load_Finnish_electricity_firm
    
    # import Finnish electricity distribution firms data
    data = load_Finnish_electricity_firm(x_select=['OPEX', 'CAPEX'],
                                          y_select=['Energy'])
    
    # First stage: solve the CNLS model
    model = CNLS.CNLS(y=data.y, x=data.x, z=None, cet = CET_ADDI, fun = FUN_PROD, rts = RTS_VRS)
    model.optimize(OPT_LOCAL)

    # Second stage: shift the residuals/intercept
    print(model.get_adjusted_residual())    
    print(model.get_adjusted_alpha())
