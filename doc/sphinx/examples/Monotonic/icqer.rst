=======================
Isotonic CQR/CER
=======================

Similarly to ICNLS, the Isotonic CQR and CER approaches are defined as follows

ICQR estimator:

.. math::
    :nowrap:

    \begin{alignat*}{2}
    \underset{\mathbf{\alpha},\mathbf{\beta },{{\mathbf{\varepsilon }}^{\text{+}}},{{\mathbf{\varepsilon }}^{-}}}{\mathop{\min }}&\,
    \tau \sum\limits_{i=1}^{n}{\varepsilon _{i}^{+}}+(1-\tau )\sum\limits_{i=1}^{n}{\varepsilon _{i}^{-}}  &{}& \\ 
    & \text{s.t.} \\
    & y_i=\mathbf{\alpha}_i+ \beta_i^{'}x_i+\varepsilon _i^{+}-\varepsilon _i^{-} &\quad& \forall i\\
    & p_{ih}(\mathbf{\alpha}_i+\beta_{i}^{'}x_i) \le p_{ih}(\mathbf{\alpha}_h+\beta _h^{'}x_i)  &{}& \forall i,h \\
    & \beta_i\ge 0 &{}& \forall i \\
    & \varepsilon _i^{+}\ge 0,\ \varepsilon_i^{-} \ge 0 &{}& \forall i
    \end{alignat*}

ICER estimator: 

.. math::
    :nowrap:

    \begin{alignat*}{2}
    \underset{\mathbf{\alpha},\mathbf{\beta},{{\mathbf{\varepsilon }}^{\text{+}}},{\mathbf{\varepsilon }}^{-}}{\mathop{\min}}&\,
    \tilde{\tau} \sum\limits_{i=1}^n(\varepsilon _i^{+})^2+(1-\tilde{\tau} )\sum\limits_{i=1}^n(\varepsilon_i^{-})^2   &{}&  \\ 
    & \text{s.t.} \\
    & y_i=\mathbf{\alpha}_i+ \beta_i^{'}x_i+\varepsilon _i^{+}-\varepsilon _i^{-} &\quad& \forall i\\
    & p_{ih}(\mathbf{\alpha}_i+\beta_{i}^{'}x_i) \le p_{ih}(\mathbf{\alpha}_h+\beta _h^{'}x_i)  &{}& \forall i,h \\
    & \beta_i\ge 0 &{}& \forall i \\
    & \varepsilon _i^{+}\ge 0,\ \varepsilon_i^{-} \ge 0 &{}& \forall i
    \end{alignat*}


Example ICQR `[.ipynb] <https://colab.research.google.com/github/ds2010/pyStoNED/blob/master/notebooks/ICQR.ipynb>`__
------------------------------------------------------------------------------------------------------------------------
    
In the following code, we estimate an additive production function using ICQR approach.
    
.. code:: python
    
        # import packages
        from pystoned import ICQER
        from pystoned.constant import CET_ADDI, FUN_PROD, OPT_LOCAL, RTS_VRS
        from pystoned.dataset import load_Finnish_electricity_firm
        
        # import Finnish electricity distribution firms data
        data = load_Finnish_electricity_firm(x_select=['OPEX', 'CAPEX'], y_select=['Energy'])
        
        # define and solve the CNLS model
        model = ICQER.ICQR(y=data.y, x=data.x, tau = 0.9, z=None, cet = CET_ADDI, fun = FUN_PROD, rts = RTS_VRS)
        model.optimize(OPT_LOCAL)
    
        # display residuals
        model.display_residual()


Example ICER `[.ipynb] <https://colab.research.google.com/github/ds2010/pyStoNED/blob/master/notebooks/ICER.ipynb>`__
-----------------------------------------------------------------------------------------------------------------------
        
We next demostrate how to estimate an additive production function using ICER approach.
        
.. code:: python
        
        # import packages
        from pystoned import ICQER
        from pystoned.constant import CET_ADDI, FUN_PROD, OPT_LOCAL, RTS_VRS
        from pystoned.dataset import load_Finnish_electricity_firm
            
        # import Finnish electricity distribution firms data
        data = load_Finnish_electricity_firm(x_select=['OPEX', 'CAPEX'], y_select=['Energy'])
            
        # define and solve the CNLS model
        model = ICQER.ICER(y=data.y, x=data.x, tau = 0.9, z=None, cet = CET_ADDI, fun = FUN_PROD, rts = RTS_VRS)
        model.optimize(OPT_LOCAL)
        
        # display residuals
        model.display_residual()