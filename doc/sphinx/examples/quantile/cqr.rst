=============================================
Convex quantile regression
=============================================

While CNLS estimates the conditional mean :math:`E(y_i |x_i)`, quantile regression aims at 
estimating the conditional median or other quantiles of the response variable
(Koenker and Bassett 1978; Koenker 2005) 
and provides an overall picture of the conditional distributions at any given quantiles :math:`\tau`. 
In this section, we extend CNLS problem to estimate convex quantile regression (CQR)
(Wang et al., 2014; Kuosmanen et al., 2015) 
and convex expectile regression (CER) (Kuosmanen et al., 2020; Dai et al., 2020; Kuosmanen and Zhou, 2021). 
Note that both quantile and expectile estimations are more robust to outliers and heteroscedasticity
than the CNLS estimation.

Given a pre-specified quantile :math:`\tau \in (0, 1)`, the CQR model is defined as 

.. math::
    :nowrap:
    
    \begin{alignat}{2}
        \underset{\alpha,\boldsymbol{\beta},{{\varepsilon}^{\text{+}}},{{\varepsilon}^{-}}}{\mathop{\min }}&\,
         \tau \sum\limits_{i=1}^{n}{\varepsilon _{i}^{+}}+(1-\tau )\sum\limits_{i=1}^{n}{\varepsilon _{i}^{-}}  &{}&  \\ 
        \textit{s.t.}\quad 
        & y_i=\alpha_i+ \boldsymbol{\beta}_i^{'}x_i+\varepsilon _i^{+} - \varepsilon _i^{-} &{}& \forall i \notag \\
        & \alpha_i+\boldsymbol{\beta}_i^{'}x_i \le \alpha_h+\boldsymbol{\beta}_h^{'}x_i  &{}& \forall i,h  \notag \\
        & \boldsymbol{\beta}_i\ge 0 &{}& \forall i \notag  \\
        & \varepsilon _i^{+}\ge 0,\ \varepsilon_i^{-} \ge 0 &{}& \forall i \notag 
    \end{alignat}

Where :math:`\varepsilon^{+}_i` and :math:`\varepsilon^{-}_i` denote the two non-negative components. 
The last set of constraints is the sign constraint of the error terms. The other constraints are the same as in the CNLS problems.


Example: quantile estimation `[.ipynb] <https://colab.research.google.com/github/ds2010/pyStoNED/blob/master/notebooks/CQR.ipynb>`_
-----------------------------------------------------------------------------------------------------------------------------------------

.. code:: python

    # import packages
    from pystoned import CQER
    from pystoned.constant import CET_ADDI, FUN_PROD, OPT_LOCAL, RTS_VRS
    from pystoned import dataset as dataset

    # import the GHG example data
    data = dataset.load_GHG_abatement_cost(x_select=['HRSN', 'CPNK', 'GHG'],
                                            y_select=['VALK'])

    # calculate the quantile model
    model = CQER.CQR(y=data.y, x=data.x, tau=0.5, z=None, 
                                                  cet=CET_ADDI, 
                                                  fun=FUN_PROD, 
                                                  rts=RTS_VRS)
    model.optimize(OPT_LOCAL)

    # display estimated alpha and beta
    model.display_alpha()
    model.display_beta() 

    # display estimated residuals
    model.display_positive_residual()
    model.display_negative_residual()