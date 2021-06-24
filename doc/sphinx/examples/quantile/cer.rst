=================================
Convex expectile regression
=================================

As shown in Kuosmanen et al (2015), the convex quantile regression may suffer from the non-uniqueness problem due to that 
Problem (2.7) is a linear programming problem. To address this problem, Kuosmanen et al (2015) purpose a convex expectile regression
(CER) approach, where a quadratic objective function is used to ensure unique estimates of the quantile functions. 

.. math::
    :nowrap:

    \begin{alignat}{2}
    \underset{\alpha,\boldsymbol{\beta},{{\varepsilon}^{\text{+}}},{\varepsilon}^{-}}{\mathop{\min}}&\,
    \tilde{\tau} \sum\limits_{i=1}^n(\varepsilon _i^{+})^2+(1-\tilde{\tau} )\sum\limits_{i=1}^n(\varepsilon_i^{-})^2   &{}&  \\ 
    \textit{s.t.}\quad 
    & y_i=\alpha_i+ \boldsymbol{\beta}_i^{'}x_i+\varepsilon _i^{+}-\varepsilon _i^{-}  &{}&  \forall i \notag  \\
    & \alpha_i+\boldsymbol{\beta}_i^{'}x_i \le \alpha_h+\boldsymbol{\beta}_h^{'}x_i  &{}& \forall i,h \notag  \\
    & \boldsymbol{\beta}_i\ge 0  &{}&  \forall i  \notag  \\
    & \varepsilon _i^{+}\ge 0,\ \varepsilon_i^{-} \ge 0  &{}& \forall i \notag 
    \end{alignat}


Example: expectile estimation `[.ipynb] <https://colab.research.google.com/github/ds2010/pyStoNED/blob/master/notebooks/CER.ipynb>`_
----------------------------------------------------------------------------------------------------------------------------------------
    
.. code:: python
    
    # import packages
    from pystoned import CQER
    from pystoned.constant import CET_ADDI, FUN_PROD, OPT_LOCAL, RTS_VRS
    from pystoned import dataset as dataset

    # import the GHG example data
    data = dataset.load_GHG_abatement_cost(x_select=['HRSN', 'CPNK', 'GHG'], y_select=['VALK'])

    # calculate the expectile model
    model = CQER.CER(y=data.y, x=data.x, tau=0.5, z=None, cet=CET_ADDI, fun=FUN_PROD, rts=RTS_VRS)
    model.optimize(OPT_LOCAL)

    # display estimated alpha and beta
    model.display_alpha()
    model.display_beta() 

    # display estimated residuals
    model.display_positive_residual()
    model.display_negative_residual() 
