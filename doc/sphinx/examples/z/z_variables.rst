========================
Contextual variables
========================

In practice, a firm’s ability to operate efficiently often depends on operational conditions and practices, 
such as the production environment and the firm specific characteristics for example 
technology  selection  or  managerial  practices.  Banker  and  Natarajan (2008) refer to both variables that 
characterize operational conditions and practices as contextual variables.

* Contextual variables are often (but not always) **external factors** that are beyond the control of firms

    - Examples: competition, regulation, weather, location
    - Need to adjust efficiency estimates for operating environment
    - Policy makers may influence the operating environment

* Contextual variables can also be **internal factors**

    - Examples: management practices, ownership
    - Better understanding of the impacts of internal factors can help the firm to improve performance


By introducing the contextual variables $z_i$, the multiplicative model (2.5)
is reformulated as an partial log-linear model to take the operational conditions and 
practices into account.

.. math::
    :nowrap:

    \begin{align}
    \ln y_i = \ln f(\boldsymbol{x_i}) + \boldsymbol{\delta}^{'}\boldsymbol{z}_i + v_i - u_i
    \end{align}
    
where parameter vector :math:`\boldsymbol{\delta}=(\delta_1...\delta_r)` represents the 
marginal effects of contextual variables on output. 
All other variables maintain their previous definitions. 
Further, we can also introduce the contextual variables to 
additive model. In this section, we take the multiplicative 
production model as our stating point.

CNLS with z variables
----------------------

Following Johnson and Kuosmanen (2011), we incorporate the contextual variables in the 
multiplicative CNLS estimation and redefine it as follows:

.. math::
    :nowrap:

    \begin{align}
    & \underset{\alpha, \boldsymbol{\beta}, \boldsymbol{\delta}, \varepsilon} {\min} \sum_{i=1}^n\varepsilon_i^2  &{}& \\
    \textit{s.t.}\quad 
    &  \ln y_i = \ln(\phi_i+1) + \boldsymbol{\delta}^{'}\boldsymbol{z}_i + \varepsilon_i  &{}&  \forall i \notag\\
    &  \phi_i  = \alpha_i + \boldsymbol{\beta}_i^{'}\boldsymbol{x}_i -1 &{}&  \forall i \notag\\
    &  \alpha_i + \boldsymbol{\beta}_i^{'}\boldsymbol{x}_i \le \alpha_j + \boldsymbol{\beta}_j^{'}\boldsymbol{x}_i  &{}&   \forall i, j \notag\\
    &  \boldsymbol{\beta}_i \ge \boldsymbol{0} &{}&   \forall i \notag
    \end{align}

Denote by :math:`\hat{\boldsymbol{\delta}}` the coefficients of the contextual variables obtained 
as the optimal solution to above nonlinear problem. Johnson and Kuosmanen (2011) examine the 
statistical properties of this estimator in detail, showing its unbiasedness, consistency, 
and asymptotic efficiency. 


CER with z variables
----------------------

Following Kuosmanen et al. (2021), we can also incorporate the contextual variable in 
the multiplicative CER estimation.

.. math::
    :nowrap:

    \begin{align}
    \underset{\phi,\alpha,\boldsymbol{\beta},{{\varepsilon}^{\text{+}}},{\varepsilon}^{-}}{\mathop{\min}}&\,
        \tilde{\tau} \sum\limits_{i=1}^n(\varepsilon _i^{+})^2+(1-\tilde{\tau} )\sum\limits_{i=1}^n(\varepsilon_i^{-})^2   &{}&  \\ 
    \textit{s.t.}\quad 
    &  \ln y_i = \ln(\phi_i+1) + \boldsymbol{\delta}^{'}\boldsymbol{z}_i + \varepsilon _i^{+}-\varepsilon _i^{-}  &{}&  \forall i \notag\\
    &  \phi_i  = \alpha_i + \boldsymbol{\beta}_i^{'}\boldsymbol{x}_i -1 &{}&  \forall i \notag\\
    &  \alpha_i + \boldsymbol{\beta}_i^{'}\boldsymbol{x}_i \le \alpha_j + \boldsymbol{\beta}_j^{'}\boldsymbol{x}_i  &{}&   \forall i, j \notag\\
    &  \boldsymbol{\beta}_i \ge \boldsymbol{0} &{}&   \forall i \notag \\
    &  \varepsilon _i^{+}\ge 0,\ \varepsilon_i^{-} \ge 0  &{}& \forall i \notag 
    \end{align}


Example `[.ipynb] <https://colab.research.google.com/github/ds2010/pyStoNED/blob/master/notebooks/StoNEZD.ipynb>`_
------------------------------------------------------------------------------------------------------------------------------

In the following code, we estimatie a log-transformed cost function model with z-variable and 
show how to obtain the firm-specific inefficiency.

.. code:: python

    # import packages
    from pystoned import CNLS, StoNED
    from pystoned.constant import CET_MULT, FUN_COST, RTS_CRS, RED_MOM
    from pystoned.dataset import load_Finnish_electricity_firm
    
    # import all data (including the contextual varibale)
    data = load_Finnish_electricity_firm(x_select=['Energy', 'Length', 'Customers'],   
                                         y_select=['TOTEX'],
                                         z_select=['PerUndGr'])

    # define and solve the StoNED model using MoM approach
    model = CNLS.CNLS(y=data.y, x=data.x, z=data.x, cet = CET_MULT, fun = FUN_COST, rts = RTS_CRS) 
    model.optimize('email@address')

    # Residual decomposition
    rd = StoNED.StoNED(model)
    print(rd.get_technical_inefficiency(RED_MOM))
