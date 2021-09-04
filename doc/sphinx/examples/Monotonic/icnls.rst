======================
Isotonic CNLS 
======================

This section introduces a variant of CNLS estimator, Isotonic Convex Nonparametric Least squares (ICNLS), 
where ICNLS only relies on the monotonic assumption. To relax the concavity assumption in CNLS estimation
(i.e., estimating a production function), we have to rephrase the Afriat inequality constraint in Problem 
（2.2). To do that, we define a binary matrix :math:`P=[p_{ij}]_{n x n}` to represent 
the isotonicity (Keshvari and Kuosmanen, 2013).

Define the binary matrix :math:`P=[p_{ij}]_{n x n}` as follows

.. math::
    :nowrap:
    
    \begin{align*}
        p_{ij} = 
        \begin{cases} 
            1       & \text{if } x_i \preccurlyeq x_j \\
            0       & \text{otherwise}
        \end{cases}
    \end{align*}

Applying enumeration method to define the elements of matrix :math:`P`, we solving the following QP problem

.. math::
    :nowrap:
    
    \begin{align*}
        & \underset{\alpha, \beta, \varepsilon} {min} \sum_{i=1}^n\varepsilon_i^2 \\
        & \text{s.t.} \\
        &  y_i = \alpha_i + \beta_i^{'}X_i + \varepsilon_i \quad \forall i \\
        &  p_{ij}(\alpha_i + \beta_i^{'}X_i) \le p_{ij}(\alpha_j + \beta_j^{'}X_i)  \quad  \forall i, j\\
        &  \beta_i \ge 0 \quad  \forall i \\
    \end{align*}

Note that the concavity constraints between units :math:`i` and :math:`j` is relaxed by the matrix :math:`P_{ij}=0`.
If the :math:`P_{ij}=1` for all :math:`i` and :math:`j`, then the above QP problem (i.e., ICNLS problem) reduces to
CNLS problem.


Example: Isotonic CNLS(ICNLS) `[.ipynb] <https://colab.research.google.com/github/ds2010/pyStoNED/blob/master/notebooks/ICNLS.ipynb>`_
-----------------------------------------------------------------------------------------------------------------

In the following code, we estimate an additive production function using ICNLS approach.

.. code:: python

    # import packages
    from pystoned import ICNLS
    from pystoned.constant import CET_ADDI, FUN_PROD, OPT_LOCAL, RTS_VRS
    from pystoned.dataset import load_Finnish_electricity_firm
    
    # import Finnish electricity distribution firms data
    data = load_Finnish_electricity_firm(x_select=['OPEX', 'CAPEX'], y_select=['Energy'])
    
    # define and solve the ICNLS model
    model = ICNLS.ICNLS(y=data.y, x=data.x, z=None, cet = CET_ADDI, fun = FUN_PROD, rts = RTS_VRS)
    model.optimize(OPT_LOCAL)

    # display residuals
    model.display_residual()
