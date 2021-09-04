================================
CQR/CER with multiple outputs
================================


Similarly to CNLS with DDF, we present another two approaches integrating DDF to convex quantile/expectile regression and we also consider modeling the undesirable outputs. 

- without undesirable outputs

    1. CQR-DDF model

    .. math::
        :nowrap:

            \begin{alignat}{2}
            \underset{\alpha,\boldsymbol{\beta},\boldsymbol{\gamma},\varepsilon^{+},\varepsilon^{-}}{\mathop{\min }}&\,
            \tau \sum\limits_{i=1}^{n}{\varepsilon _{i}^{+}}+(1-\tau )\sum\limits_{i=1}^{n}{\varepsilon _{i}^{-}}  &{\quad}& \\ 
            \textit{s.t.}\quad 
            &  \boldsymbol{\gamma}_i^{'}\boldsymbol{y}_i = \alpha_i + \boldsymbol{\beta}_i^{'}\boldsymbol{x}_i + \varepsilon^+_i - \varepsilon^-_i &{\quad}& \forall i \notag \\
            &  \alpha_i + \boldsymbol{\beta}_i^{'}\boldsymbol{x}_i -\boldsymbol{\gamma}_i^{'}\boldsymbol{y}_i \le \alpha_j + \boldsymbol{\beta}_j^{'}\boldsymbol{x}_i -\boldsymbol{\gamma}_j^{'}\boldsymbol{y}_i &{\quad}&  \forall i, j \notag \\
            &  \boldsymbol{\gamma}_i^{'} g^{y}  + \boldsymbol{\beta}_i^{'} g^{x}  = 1  &{\quad}& \forall i \notag \\ 
            &  \boldsymbol{\beta}_i \ge \boldsymbol{0} , \boldsymbol{\gamma}_i \ge \boldsymbol{0} &{\quad}&  \forall i \notag \\
            & \varepsilon _i^{+}\ge 0,\ \varepsilon_i^{-} \ge 0 &{\quad}& \forall i \notag
            \end{alignat}
    
    2. CER-DDF model

    .. math::
        :nowrap:

            \begin{alignat}{2}
            \underset{\alpha,\boldsymbol{\beta},\boldsymbol{\gamma},\varepsilon^{+},\varepsilon^{-}}{\mathop{\min}}&\,
            \tilde{\tau} \sum\limits_{i=1}^n(\varepsilon _i^{+})^2+(1-\tilde{\tau} )\sum\limits_{i=1}^n(\varepsilon_i^{-})^2   &{\quad}&  \\ 
            \textit{s.t.}\quad 
            &  \boldsymbol{\gamma}_i^{'}\boldsymbol{y}_i = \alpha_i + \boldsymbol{\beta}_i^{'}\boldsymbol{x}_i + \varepsilon^+_i - \varepsilon^-_i &{\quad}& \forall i \notag \\
            &  \alpha_i + \boldsymbol{\beta}_i^{'}\boldsymbol{x}_i -\boldsymbol{\gamma}_i^{'}\boldsymbol{y}_i \le \alpha_j + \boldsymbol{\beta}_j^{'}\boldsymbol{x}_i -\boldsymbol{\gamma}_j^{'}\boldsymbol{y}_i &{\quad}&  \forall i, j \notag \\
            &  \boldsymbol{\gamma}_i^{'} g^{y}  + \boldsymbol{\beta}_i^{'} g^{x}  = 1  &{\quad}& \forall i \notag \\ 
            &  \boldsymbol{\beta}_i \ge \boldsymbol{0} , \boldsymbol{\gamma}_i \ge \boldsymbol{0} &{\quad}& \forall i  \notag \\
            & \varepsilon _i^{+}\ge 0,\ \varepsilon_i^{-} \ge 0 &{\quad}& \forall i \notag
            \end{alignat}


- with undesirable outputs

    1. CQR-DDF-b model

    .. math::
        :nowrap:

            \begin{alignat}{2}
            \underset{\alpha,\boldsymbol{\beta},\boldsymbol{\gamma},\boldsymbol{\delta}, \varepsilon^{+},\varepsilon^{-}}{\mathop{\min }}&\,
            \tau \sum\limits_{i=1}^{n}{\varepsilon _{i}^{+}}+(1-\tau )\sum\limits_{i=1}^{n}{\varepsilon _{i}^{-}}  &{\quad}& \\ 
            \textit{s.t.}\quad 
            &  \boldsymbol{\gamma}_i^{'}\boldsymbol{y_i} = \alpha_i + \boldsymbol{\beta}_i^{'}\boldsymbol{x_i} + \boldsymbol{\delta}_i^{'}\boldsymbol{b}_i + \varepsilon^+_i - \varepsilon^-_i &{\quad}& \forall i \notag \\
            &  \alpha_i + \boldsymbol{\beta}_i^{'}\boldsymbol{x_i} + \boldsymbol{\delta}_i^{'}\boldsymbol{b}_i -\boldsymbol{\gamma}_i^{'}\boldsymbol{y_i} \le \alpha_j + \boldsymbol{\beta}_j^{'}\boldsymbol{x_i} + \boldsymbol{\delta}_j^{'}\boldsymbol{b}_i -\boldsymbol{\gamma}_j^{'}\boldsymbol{y_i} &{\quad}& \forall i, j \notag \\
            &  \boldsymbol{\gamma}_i^{'} g^{y}  + \boldsymbol{\beta}_i^{'} g^{x} + \boldsymbol{\delta}_i^{'}g^{b} = 1 &{\quad}& \forall i \notag \\ 
            &  \boldsymbol{\beta}_i \ge \boldsymbol{0}, \delta_i \ge \boldsymbol{0}, \boldsymbol{\gamma}_i \ge \boldsymbol{0} &{\quad}&  \forall i \notag \\
            & \varepsilon _i^{+}\ge 0,\ \varepsilon_i^{-} \ge 0 &{\quad}& \forall i \notag
            \end{alignat}

    2. CER-DDF-b model  

    .. math::
        :nowrap:

            \begin{alignat}{2}
            \underset{\alpha,\boldsymbol{\beta}, \boldsymbol{\gamma},\boldsymbol{\delta}, \varepsilon^{+},\varepsilon^{-}}{\mathop{\min}}&\,
            \tilde{\tau} \sum\limits_{i=1}^n(\varepsilon _i^{+})^2+(1-\tilde{\tau} )\sum\limits_{i=1}^n(\varepsilon_i^{-})^2    &{\quad}&  \\ 
            \textit{s.t.}\quad 
            &  \boldsymbol{\gamma}_i^{'}\boldsymbol{y_i} = \alpha_i + \boldsymbol{\beta}_i^{'}\boldsymbol{x_i} + \boldsymbol{\delta}_i^{'}\boldsymbol{b}_i + \varepsilon^+_i - \varepsilon^-_i  &{\quad}& \forall i \notag \\
            &  \alpha_i + \boldsymbol{\beta}_i^{'}\boldsymbol{x_i} + \boldsymbol{\delta}_i^{'}\boldsymbol{b}_i -\boldsymbol{\gamma}_i^{'}\boldsymbol{y_i} \le \alpha_j + \boldsymbol{\beta}_j^{'}\boldsymbol{x_i} + \boldsymbol{\delta}_j^{'}\boldsymbol{b}_i -\boldsymbol{\gamma}_j^{'}\boldsymbol{y_i}  &{\quad}&  \forall i, j  \notag \\
            &  \boldsymbol{\gamma}_i^{'} g^{y}  + \boldsymbol{\beta}_i^{'} g^{x} + \boldsymbol{\delta}_i^{'}g^{b} = 1   &{\quad}& \forall i \notag \\ 
            &  \boldsymbol{\beta}_i \ge \boldsymbol{0}, \delta_i \ge \boldsymbol{0}, \boldsymbol{\gamma}_i \ge \boldsymbol{0}  &{\quad}&  \forall i  \notag \\
            & \varepsilon _i^{+}\ge 0,\ \varepsilon_i^{-} \ge 0  &{\quad}& \forall i \notag
            \end{alignat}


Example: CQR-DDF `[.ipynb] <https://colab.research.google.com/github/ds2010/pyStoNED/blob/master/notebooks/CQR_DDF.ipynb>`__
-------------------------------------------------------------------------------------------------------------------------------
    
.. code:: python
    
        # import packages
        from pystoned import CQERDDF
        from pystoned.constant import FUN_PROD, OPT_LOCAL
        from pystoned import dataset as dataset
        
        # import Finnish electricity distribution firms data
        data = dataset.load_Finnish_electricity_firm(x_select=['OPEX', 'CAPEX'],
                                            y_select=['Energy', 'Length', 'Customers'])
        
        # define and solve the CQR-DDF model
        model = CQERDDF.CQRDDF(y=data.y, x=data.x, b=None, tau=0.9, fun = FUN_PROD, gx= [1.0, 0.0], gb=None, gy= [0.0, 0.0, 0.0])
        model.optimize(OPT_LOCAL)
    
        # display the residual
        model.display_positive_residual()
        model.display_negative_residual()


Example: CQR-DDF with bad output `[.ipynb] <https://colab.research.google.com/github/ds2010/pyStoNED/blob/master/notebooks/CQR_DDF-b.ipynb>`__
------------------------------------------------------------------------------------------------------------------------------------------------
            
.. code:: python
            
        # import packages
        from pystoned import CQERDDF
        from pystoned.constant import FUN_PROD, OPT_LOCAL
        from pystoned import dataset as dataset
                
        # import the GHG emissions data
        data = dataset.load_GHG_abatement_cost()
                
        # define and solve the CQR-DDF model (with undesirable outputs)
        model = CQERDDF.CQRDDF(y=data.y, x=data.x, b=data.b, tau=0.9, fun = FUN_PROD, gx= [0.0, 0.0], gb=[-1], gy=[1])
        model.optimize(OPT_LOCAL)
            
        # display the delta
        model.display_delta()