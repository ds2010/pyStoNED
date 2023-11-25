============================
CNLS with multiple outputs
============================

Until now, the convex regression approaches have been presented in the single output, 
multiple input setting. In this section, we describe the CNLS/CQR/CER approaches 
within the directional distance function (DDF) framework to handle with multiple 
input-multiple output data (Chambers et al., 1996, 1998). 

Consider the following quadratic programming problem (Kuosmanen and Johnson, 2017)

.. math::
    :nowrap:

    \begin{alignat}{2}
    \underset{\alpha, \boldsymbol{\beta}, \boldsymbol{\gamma}, \varepsilon}{\mathop{\min}}&\sum_{i=1}^n\varepsilon_i^2  &{\quad}&  \\
    \textit{s.t.}\quad 
    &  \boldsymbol{\gamma}_i^{'}\boldsymbol{y}_i = \alpha_i + \beta_i^{'}\boldsymbol{x}_i - \varepsilon_i &{\quad}& \forall i \notag \\
    &  \alpha_i + \boldsymbol{\beta}_i^{'}\boldsymbol{x}_i -\boldsymbol{\gamma}_i^{'}\boldsymbol{y}_i \le \alpha_j + \boldsymbol{\beta}_j^{'}\boldsymbol{x}_i -\boldsymbol{\gamma}_j^{'}\boldsymbol{y}_i  &{\quad}&  \forall i, j \notag\\
    &  \boldsymbol{\gamma}_i^{'} g^{y}  + \boldsymbol{\beta}_i^{'} g^{x}  = 1  &{\quad}& \forall i  \notag \\ 
    &  \boldsymbol{\beta}_i \ge \boldsymbol{0}, \boldsymbol{\gamma}_i \ge \boldsymbol{0} &{\quad}& \forall i \notag
    \end{alignat}

where the residual :math:`\varepsilon_i` represents the estimated value of :math:`d` (:math:`\vec{D}(x_i,y_i,g^x,g^y)+u_i`). 
Besides the same notations as the CNLS estimator, we also introduce new firm-specific coefficients :math:`\boldsymbol{\gamma_i}`
that represent marginal effects of outputs to the DDF.

The first constraint defines the distance to the frontier as a linear function of inputs and outputs. 
The linear approximation of the frontier is based on the tangent hyperplanes, analogous to the original 
CNLS formulation. The second set of constraints is the system of Afriat inequalities that impose global 
concavity. The third constraint is a normalization constraint that ensures the translation property. 
The last two constraints impose monotonicity in all inputs and outputs. It is straightforward to show 
that the CNLS estimator of function $d$ satisfies the axioms of free disposability, convexity, and the translation property.


Example: CNLS-DDF `[.ipynb] <https://colab.research.google.com/github/ds2010/pyStoNED/blob/master/notebooks/DDF_withoutUndesirableOutput.ipynb>`__
----------------------------------------------------------------------------------------------------------------------------------------------------

.. code:: python

    # import packages
    from pystoned import CNLSDDF
    from pystoned.constant import FUN_PROD, OPT_LOCAL
    from pystoned.dataset import load_Finnish_electricity_firm
    
    # import Finnish electricity distribution firms data
    data = load_Finnish_electricity_firm(x_select=['OPEX', 'CAPEX'],
                                        y_select=['Energy', 'Length', 'Customers'])
    
    # define and solve the CNLS-DDF model
    model = CNLSDDF.CNLSDDF(y=data.y, x=data.x, b=None, fun = FUN_PROD, gx= [1.0, 0.0], gb=None, gy= [0.0, 0.0, 0.0])
    model.optimize(OPT_LOCAL)

    # display the estimates (alpha, beta, gamma, and residual)
    model.display_alpha()
    model.display_beta()
    model.display_gamma()
    model.display_residual()
