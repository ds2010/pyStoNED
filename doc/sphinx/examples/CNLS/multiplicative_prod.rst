=============================================
Production function: multiplicative model
=============================================

Most SFA studies use Cobb-Douglas or translog functional forms where inefficiency and 
noise affect production in a multiplicative fashion. Note that the assumption of 
constant returns to scale (CRS) would also require multiplicative error structure. 

In the context of VRS, the log-transformed ``CNLS`` formulation:

.. math::
    :nowrap:

    \begin{align*}
        & \underset{\alpha, \beta, \varepsilon} {min} \sum_{i=1}^n\varepsilon_i^2 \\
        & \text{s.t.} \\
        &  \text{ln}y_i = \text{ln}(\phi_i+1) + \varepsilon_i  \quad \forall i\\
        & \phi_i  = \alpha_i+\beta_i^{'}X_i -1 \quad \forall i \\
        &  \alpha_i + \beta_i^{'}X_i \le \alpha_j + \beta_j^{'}X_i  \quad  \forall i, j\\
        &  \beta_i \ge 0 \quad  \forall i \\
    \end{align*}

The CRS log-transformed ``CNLS`` formulation:

.. math::
    :nowrap:
    
    \begin{align*}
        & \underset{\alpha, \beta, \varepsilon} {min} \sum_{i=1}^n\varepsilon_i^2 \\
        & \text{s.t.} \\
        &  \text{ln}y_i = \text{ln}(\phi_i+1) + \varepsilon_i  \quad \forall i\\
        &   \phi_i  = \beta_i^{'}X_i -1 \quad \forall i \\
        &  \beta_i^{'}X_i \le \beta_j^{'}X_i  \quad  \forall i, j\\
        &  \beta_i \ge 0 \quad  \forall i \\
    \end{align*}

where :math:`\phi_i+1` is the CNLS estimator of :math:`E(y_i|x_i)`. The value of one is added here 
to make sure that the computational algorithms do not try to take logarithm of zero. 
The first equality can be interpreted as the log transformed regression equation 
(using the natural logarithm function :math:`ln(.)`). The rest of constraints 
are similar to additive production function model. The use of :math:`\phi_i` allows
the estimation of a multiplicative relationship between output and 
input while assuring convexity of the production possibility set in original 
input-output space.

Example `[.ipynb] <https://colab.research.google.com/github/ds2010/pyStoNED/blob/master/notebooks/CNLS_mult_prod.ipynb>`_
----------------------------------------------------------------------------------------------------------------------------------

In the following code, we estimate two multiplicative production functions with pyStoNED.

.. code:: python

    # import packages
    from pystoned import CNLS
    from pystoned.constant import CET_MULT, FUN_PROD, FUN_COST, OPT_LOCAL, RTS_VRS, RTS_CRS
    from pystoned.dataset import load_Finnish_electricity_firm
    
    # import Finnish electricity distribution firms data
    data = load_Finnish_electricity_firm(x_select=['OPEX', 'CAPEX'],
                                        y_select=['Energy'])

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
