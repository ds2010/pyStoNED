========================
CER with Z variables
========================

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

Example: CER-Z `[.ipynb] <https://colab.research.google.com/github/ds2010/pyStoNED/blob/master/notebooks/CER_Z.ipynb>`_
------------------------------------------------------------------------------------------------------------------------
        
.. code:: python
    
    # import packages
    from pystoned import CQER
    from pystoned.constant import CET_MULT, FUN_COST, RTS_CRS, RED_MOM
    from pystoned.dataset import load_Finnish_electricity_firm
    
    # import all data (including the contextual varibale)
    data = load_Finnish_electricity_firm(x_select=['Energy', 'Length', 'Customers'],   
                                         y_select=['TOTEX'],
                                         z_select=['PerUndGr'])

    # define and solve the CER-Z model
    model = CQER.CER(y=data.y, x=data.x, z=data.z, tau=0.5, cet = CET_MULT, fun = FUN_COST, rts = RTS_CRS) 
    model.optimize('email@address')

    # display the coefficient of contextual variable
    model.display_lamda()