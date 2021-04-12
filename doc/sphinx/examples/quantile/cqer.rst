=============================================
Convex quantile and expectile regression
=============================================

While CNLS estimates the conditional mean :math:`E(y_i |x_i)`, quantile regression aims at 
estimating the conditional median or other quantiles of the response variable 
(Koenker and Bassett 1978; Koenker 2005). Denoting the pre-assigned quantile 
by parameter :math:`\tau \in (0, 1)`, we can modify the CNLS problem to estimate 
convex quantile regression (CQR) (Wang et al. 2014, Kuosmanen et al. 2015) as follows

.. math::
    :nowrap:
    
    \begin{alignat*}{2}
    \underset{\mathbf{\alpha},\mathbf{\beta },{{\mathbf{\varepsilon }}^{\text{+}}},{{\mathbf{\varepsilon }}^{-}}}{\mathop{\min }}&\,
    \tau \sum\limits_{i=1}^{n}{\varepsilon _{i}^{+}}+(1-\tau )\sum\limits_{i=1}^{n}{\varepsilon _{i}^{-}}  &{}& \\ 
    & \text{s.t.} \\
    & y_i=\mathbf{\alpha}_i+ \beta_i^{'}x_i+\varepsilon _i^{+}-\varepsilon _i^{-} &\quad& \forall i\\
    & \mathbf{\alpha}_i+\beta_{i}^{'}x_i \le \mathbf{\alpha}_h+\beta _h^{'}x_i  &{}& \forall i,h \\
    & \beta_i\ge 0 &{}& \forall i \\
    & \varepsilon _i^{+}\ge 0,\ \varepsilon_i^{-} \ge 0 &{}& \forall i
    \end{alignat*}

Where :math:`\varepsilon^{+}_i` and :math:`\varepsilon^{-}_i` denotes the two non-negative components. The last set of constraints is the sign 
constraint ofthe error terms. As shown in Kuosmanen et al. 2015, the convex quantile regression suffer from the non-uniqueness problem. 
To address this proplem Kuosmanen et al. 2015 purposed the convex expectile regression approach, where a quadratic objective function is 
used to ensure unique estimates of the quantile functions. 

.. math::
    :nowrap:

    \begin{alignat*}{2}
    \underset{\mathbf{\alpha},\mathbf{\beta},{{\mathbf{\varepsilon }}^{\text{+}}},{\mathbf{\varepsilon }}^{-}}{\mathop{\min}}&\,
    \tilde{\tau} \sum\limits_{i=1}^n(\varepsilon _i^{+})^2+(1-\tilde{\tau} )\sum\limits_{i=1}^n(\varepsilon_i^{-})^2   &{}&  \\ 
    & \text{s.t.} \\
    & y_i=\mathbf{\alpha}_i+ \beta_i^{'}x_i+\varepsilon _i^{+}-\varepsilon _i^{-} &\quad& \forall i\\
    & \mathbf{\alpha}_i+\beta_{i}^{'}x_i \le \mathbf{\alpha}_h+\beta _h^{'}x_i  &{}& \forall i,h \\
    & \beta_i\ge 0 &{}& \forall i \\
    & \varepsilon _i^{+}\ge 0,\ \varepsilon_i^{-} \ge 0 &{}& \forall i
    \end{alignat*}


Example `[.ipynb] <https://colab.research.google.com/github/ds2010/pyStoNED/blob/master/notebooks/CQR.ipynb>`_
-------------------------------------------------------------------------------------------------------------------

In the following code, we first estimate an convex quantile regression model with pyStoNED.

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


    # display estiamted alpha and beta
    model.display_alpha()
    model.display_beta() 

    # display estimated residuals
    model.display_positive()
    model.display_negative() 


Example `[.ipynb] <https://colab.research.google.com/github/ds2010/pyStoNED/blob/master/notebooks/CER.ipynb>`_
--------------------------------------------------------------------------------------------------------------------
    
We estimate an convex expectile regression model with pyStoNED.
    
.. code:: python
    
    # import packages
    from pystoned import CQER
    from pystoned.constant import CET_ADDI, FUN_PROD, OPT_LOCAL, RTS_VRS
    from pystoned import dataset as dataset

    # import the GHG example data
    data = dataset.load_GHG_abatement_cost(x_select=['HRSN', 'CPNK', 'GHG'], 
                                            y_select=['VALK'])

    # calculate the expectile model
    model = CQER.CER(y=data.y, x=data.x, tau=0.5, z=None, 
                                                  cet=CET_ADDI, 
                                                  fun=FUN_PROD, 
                                                  rts=RTS_VRS)
    model.optimize(OPT_LOCAL)


    # display estiamted alpha and beta
    model.display_alpha()
    model.display_beta() 

    # display estimated residuals
    model.display_positive()
    model.display_negative()     
    