================================
CQR/CER with multiple outputs
================================


Similarly to CNLS with DDF, we present another two aprroaches intergrating DDF to convex quantile/expectile regression 
and we also consider modeling the undesirable outputs. 

1. Without undesirable outputs:

1) CQR-DDF model

.. math::
    :nowrap:

    \begin{align*}
         \underset{\mathbf{\alpha},\mathbf{\beta },{{\mathbf{\varepsilon }}^{\text{+}}},{{\mathbf{\varepsilon }}^{-}}}{\mathop{\min }}&\,
    \tau \sum\limits_{i=1}^{n}{\varepsilon _{i}^{+}}+(1-\tau )\sum\limits_{i=1}^{n}{\varepsilon _{i}^{-}}  &{}& \\ 
        \text{s.t.} \\
        &  \gamma_i^{'}y_i = \alpha_i + \beta_i^{'}X_i + \varepsilon^+_i - \varepsilon^-_i \quad \forall i \\
        &  \alpha_i + \beta_i^{'}X_i -\gamma_i^{'}y_i \le \alpha_j + \beta_j^{'}X_i -\gamma_j^{'}y_i \quad  \forall i, j\\
        &  \gamma_i^{'} g^{y}  + \beta_i^{'} g^{x}  = 1  \quad \forall i \\ 
        &  \beta_i \ge 0 , \gamma_i \ge 0 \quad  \forall i \\
        & \varepsilon _i^{+}\ge 0,\ \varepsilon_i^{-} \ge 0 \quad \forall i
    \end{align*}



2) CER-DDF model

.. math::
    :nowrap:

    \begin{align*}
        \underset{\mathbf{\alpha},\mathbf{\beta},{{\mathbf{\varepsilon }}^{\text{+}}},{\mathbf{\varepsilon }}^{-}}{\mathop{\min}}&\,
        \tilde{\tau} \sum\limits_{i=1}^n(\varepsilon _i^{+})^2+(1-\tilde{\tau} )\sum\limits_{i=1}^n(\varepsilon_i^{-})^2   &{}&  \\ 
        & \text{s.t.} \\
        &  \gamma_i^{'}y_i = \alpha_i + \beta_i^{'}X_i + \varepsilon^+_i - \varepsilon^-_i \quad \forall i \\
        &  \alpha_i + \beta_i^{'}X_i -\gamma_i^{'}y_i \le \alpha_j + \beta_j^{'}X_i -\gamma_j^{'}y_i \quad  \forall i, j\\
        &  \gamma_i^{'} g^{y}  + \beta_i^{'} g^{x}  = 1  \quad \forall i \\ 
        &  \beta_i \ge 0 , \gamma_i \ge 0 \quad  \forall i \\
        & \varepsilon _i^{+}\ge 0,\ \varepsilon_i^{-} \ge 0 \quad \forall i
    \end{align*}


2. With undesirable outputs

1) CQR-DDF model

.. math::
    :nowrap:

    \begin{align*}
        \underset{\mathbf{\alpha},\mathbf{\beta },{{\mathbf{\varepsilon }}^{\text{+}}},{{\mathbf{\varepsilon }}^{-}}}{\mathop{\min }}&\,
        \tau \sum\limits_{i=1}^{n}{\varepsilon _{i}^{+}}+(1-\tau )\sum\limits_{i=1}^{n}{\varepsilon _{i}^{-}}  &{}& \\ 
        \text{s.t.} \\
        &  \gamma_i^{'}y_i = \alpha_i + \beta_i^{'}X_i + \delta_i^{'}b_i + \varepsilon^+_i - \varepsilon^-_i \quad \forall i \\
        &  \alpha_i + \beta_i^{'}X_i + \delta_i^{'}b_i -\gamma_i^{'}y_i \le \alpha_j + \beta_j^{'}X_i + \delta_j^{'}b_i -\gamma_j^{'}y_i \quad  \forall i, j\\
        &  \gamma_i^{'} g^{y}  + \beta_i^{'} g^{x} + \delta_i^{'}g^{b} = 1  \quad \forall i \\ 
        &  \beta_i \ge 0, \delta_i \ge 0, \gamma_i \ge 0 \quad  \forall i \\
        & \varepsilon _i^{+}\ge 0,\ \varepsilon_i^{-} \ge 0 \quad \forall i
    \end{align*}


2) CER-DDF model   

.. math::
    :nowrap:

    \begin{align*}
        \underset{\mathbf{\alpha},\mathbf{\beta},{{\mathbf{\varepsilon }}^{\text{+}}},{\mathbf{\varepsilon }}^{-}}{\mathop{\min}}&\,
        \tilde{\tau} \sum\limits_{i=1}^n(\varepsilon _i^{+})^2+(1-\tilde{\tau} )\sum\limits_{i=1}^n(\varepsilon_i^{-})^2   &{}&  \\ 
        & \text{s.t.} \\
        &  \gamma_i^{'}y_i = \alpha_i + \beta_i^{'}X_i + \delta_i^{'}b_i + \varepsilon^+_i - \varepsilon^-_i \quad \forall i \\
        &  \alpha_i + \beta_i^{'}X_i + \delta_i^{'}b_i -\gamma_i^{'}y_i \le \alpha_j + \beta_j^{'}X_i + \delta_j^{'}b_i -\gamma_j^{'}y_i \quad  \forall i, j\\
        &  \gamma_i^{'} g^{y}  + \beta_i^{'} g^{x} + \delta_i^{'}g^{b} = 1  \quad \forall i \\ 
        &  \beta_i \ge 0, \delta_i \ge 0, \gamma_i \ge 0 \quad  \forall i \\
        & \varepsilon _i^{+}\ge 0,\ \varepsilon_i^{-} \ge 0 \quad \forall i
    \end{align*}



Example: CER-DDF `[.ipynb] <https://colab.research.google.com/github/ds2010/pyStoNED/blob/master/sources/notebooks/CQR_DDF.ipynb>`_
--------------------------------------------------------------------------------------------------------------------------------------------------------
    
.. code:: python
    
        # import packages
        from pystoned import CNLSDDF
        from pystoned.constant import FUN_PROD, OPT_LOCAL
        from pystoned import dataset as dataset
        
        # import Finnish electricity distribution firms data
        data = dataset.load_Finnish_electricity_firm(x_select=['OPEX', 'CAPEX'],
                                            y_select=['Energy', 'Length', 'Customers'])
        
        # define and solve the CQR-DDF model
        model = CQERDDF.CQRDDF(y=data.y, x=data.x, b=None, tau=0.9, fun = FUN_PROD, gx= [1.0, 0.0], gb=None, gy= [0.0, 0.0, 0.0])
        model.optimize(OPT_LOCAL)
    
        # display the residual
        model.display_residual()


Example: CQR-DDF `[.ipynb] <https://colab.research.google.com/github/ds2010/pyStoNED/blob/master/sources/notebooks/CER_DDF.ipynb>`_
--------------------------------------------------------------------------------------------------------------------------------------
            
.. code:: python
            
        # import packages
        from pystoned import CNLSDDF
        from pystoned.constant import FUN_PROD, OPT_LOCAL
        from pystoned import dataset as dataset
                
        # import the GHG emissions data
        data = dataset.load_GHG_abatement_cost()
                
        # define and solve the CQR-DDF model (with undesirable outputs)
        model = CQERDDF.CQRDDF(y=data.y, x=data.x, b=data.b, tau=0.9, fun = FUN_PROD, gx= [0.0, 0.0], gb=[-1], gy=[1])
        model.optimize(OPT_LOCAL)
            
        # display the residual
        model.display_residual()