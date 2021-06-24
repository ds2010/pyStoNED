==================================
Multiplicative CNLS model
==================================

Similar to the most existing SFA literature, the Cobb-Douglas and translog functions are commonly assumed to be the 
functional form for the function $f$, where inefficiency $u$ and noise $v$ affect production in a multiplicative fashion.
We, thus, further consider the multiplicative specification in the nonparametric models. Note that the assumption of 
CRS would also require multiplicative error structure. The multiplicative composite error structure CNLS model is rephrased as

.. math::
    :nowrap:

    \begin{align}
        y_i = f(\boldsymbol{x}_i) \cdot \exp{(\varepsilon_i)} = f(\boldsymbol{x}_i) \cdot \exp{( v_i - u_i)} 
    \end{align}

Applying the log-transformation to Eq.\eqref{eq:eq4}, we obtain

.. math::
    :nowrap:
    
    \begin{align}
        \ln y_i = \ln f(\boldsymbol{x}_i) + v_i - u_i 
    \end{align}

To estimate the Eq.\eqref{eq:eq5}, we reformulate the additive production model 
\eqref{eq:eq2} and obtain the following log-transformed CNLS formulation:

.. math::
    :nowrap:
    
    \begin{align*}
        \underset{\alpha, \boldsymbol{\beta}, \varepsilon} \min & \sum_{i=1}^n\varepsilon_i^2  &{\quad}&\\
        \textit{s.t.}\quad 
        &  \ln y_i = \ln(\boldsymbol{\phi}_i+1) + \varepsilon_i  &{\quad}& \forall i  \notag\\
        & \boldsymbol{\phi}_i  = \alpha_i+\boldsymbol{\beta}_i^{'}\boldsymbol{x}_i -1 &{\quad}& \forall i  \notag \\
        &  \alpha_i + \boldsymbol{\beta}_i^{'}\boldsymbol{x}_i \le \alpha_j + \boldsymbol{\beta}_j^{'}\boldsymbol{x}_i  &{\quad}&  \forall i, j  \notag\\
        &  \boldsymbol{\beta}_i \ge 0 &{\quad}&  \forall i  \notag 
    \end{align*}

where :math:`\phi_i+1` is the CNLS estimator of :math:`E[y_i \, | \, x_i]`. 
The value of one is added here to make sure that the computational 
algorithms do not try to take logarithm of zero. The first equality 
can be interpreted as the log transformed regression equation (using 
the natural logarithm function :math:`\ln(.)`). The rest of constraints are 
similar to additive models. The use of :math:`\phi_i` allows the estimation 
of a multiplicative relationship between output and input while assuring 
convexity of the production possibility set in original input-output space. 
Note that one could not apply the log transformation directly to the input data 
:math:`\boldsymbol{x}` due to the fact that the piece-wise log-linear frontier does not satisfy 
the axiomatic property (i.e., concavity or convexity) of function :math:`f`.



Example `[.ipynb] <https://colab.research.google.com/github/ds2010/pyStoNED/blob/master/notebooks/CNLS_mult_prod.ipynb>`_
----------------------------------------------------------------------------------------------------------------------------------

.. code:: python

    # import packages
    from pystoned import CNLS
    from pystoned.constant import CET_MULT, FUN_PROD, FUN_COST, OPT_LOCAL, RTS_VRS, RTS_CRS
    from pystoned.dataset import load_Finnish_electricity_firm
    
    # import Finnish electricity distribution firms data
    data = load_Finnish_electricity_firm(x_select=['OPEX', 'CAPEX'], y_select=['Energy'])

    # define and solve the Multiplicative CNLS_vrs model
    model1 = CNLS.CNLS(y=data.y, x=data.x, z=None, cet = CET_MULT, fun = FUN_PROD, rts = RTS_VRS)
    model1.optimize('email@address')

    # define and solve the Multiplicative CNLS_crs model
    model2 = CNLS.CNLS(y=data.y, x=data.x, z=None, cet = CET_MULT, fun = FUN_PROD, rts = RTS_CRS)
    model2.optimize('email@address')

    # print residuals in the VRS model
    print(model1.display_residual())

    # print residuals in the CRS model
    print(model2.display_residual())


Example `[.ipynb] <https://colab.research.google.com/github/ds2010/pyStoNED/blob/master/notebooks/CNLS_mult_cost.ipynb>`_
-----------------------------------------------------------------------------------------------------------------------------------

In the following code, we estimate two multiplicative cost functions with pyStoNED.

.. code:: python

    # import packages
    from pystoned import CNLS
    from pystoned.constant import CET_MULT, FUN_PROD, FUN_COST, OPT_LOCAL, RTS_VRS, RTS_CRS
    from pystoned.dataset import load_Finnish_electricity_firm
    
    # import Finnish electricity distribution firms data
    data = load_Finnish_electricity_firm(x_select=['Energy', 'Length', 'Customers'],
                                        y_select=['TOTEX'])
    
    # define and solve the Multiplicative CNLS_vrs model
    model1 = CNLS.CNLS(y=data.y, x=data.x, z=None, cet = CET_MULT, fun = FUN_COST, rts = RTS_VRS)
    model1.optimize('email@address')

    # define and solve the Multiplicative CNLS_crs model
    model2 = CNLS.CNLS(y=data.y, x=data.x, z=None, cet = CET_MULT, fun = FUN_COST, rts = RTS_CRS)
    model2.optimize('email@address')

    # print residuals in the VRS model
    print(model1.display_residual())

    # print residuals in the CRS model
    print(model2.display_residual())