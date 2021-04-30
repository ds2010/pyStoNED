============================
CNLS with multiple outputs
============================

Until now, the CNLS/StoNED framework has been presented in the single output, 
multiple input setting. In this part we describe the CNLS estimator 
within the directional distance function (DDF) framework, Chambers et al. (1996,1998).

Consider the following QP problem

.. math::
    :nowrap:

    \begin{align*}
        & \underset{\alpha, \beta, \varepsilon} {min} \sum_{i=1}^n\varepsilon_i^2 \\
        & \text{s.t.} \\
        &  \gamma_i^{'}y_i = \alpha_i + \beta_i^{'}X_i - \varepsilon_i \quad \forall i \\
        &  \alpha_i + \beta_i^{'}X_i -\gamma_i^{'}y_i \le \alpha_j + \beta_j^{'}X_i -\gamma_j^{'}y_i \quad  \forall i, j\\
        &  \gamma_i^{'} g^{y}  + \beta_i^{'} g^{x}  = 1  \quad \forall i \\ 
        &  \beta_i \ge 0 , \gamma_i \ge 0 \quad  \forall i \\
    \end{align*}

Here the residual :math:`\hat{\varepsilon}_i` represents the estimated value of `d`
(:math:`\vec{D}(x_i,y_i,g^x,g^y)+u_i`). We also introduce new firm-specific coefficients
:math:`\gamma_i` that represent marginal effects of outputs to the DDF. 
The first constraint defines the distance to the frontier as a linear function of inputs 
and outputs. The linear approximation of the frontier is based on the tangent hyperplanes, 
analogous to the original CNLS formulation. The second set of constraints is the 
system of Afriat inequalities thatimpose global concavity. The third constraint 
is a normalization constraint that ensures the translation property. The last 
two constraints impose monotonicity in allinputs and outputs. It is straightforward 
to show that the CNLS estimator of function `d` satisfies the axioms of free disposability, 
convexity, and the translation property.


When considering undesirable outputs, the above CNLS-DDF problem can be reformulated as

.. math::
    :nowrap:

    \begin{align*}
        & \underset{\alpha, \beta, \varepsilon} {min} \sum_{i=1}^n\varepsilon_i^2 \\
        & \text{s.t.} \\
        &  \gamma_i^{'}y_i = \alpha_i + \beta_i^{'}X_i + \delta_i^{'}b_i - \varepsilon_i \quad \forall i \\
        &  \alpha_i + \beta_i^{'}X_i + \delta_i^{'}b_i -\gamma_i^{'}y_i \le \alpha_j + \beta_j^{'}X_i + \delta_j^{'}b_i -\gamma_j^{'}y_i \quad  \forall i, j\\
        &  \gamma_i^{'} g^{y}  + \beta_i^{'} g^{x} + \delta_i^{'}g^{b} = 1  \quad \forall i \\ 
        &  \beta_i \ge 0, \delta_i \ge 0, \gamma_i \ge 0 \quad  \forall i \\
    \end{align*}



Example #1 `[.ipynb] <https://colab.research.google.com/github/ds2010/pyStoNED/blob/master/notebooks/DDF_withoutUndesirableOutput.ipynb>`_
-----------------------------------------------------------------------------------------------------------------------------------------------------

In the following code, we first estimate the DDF without considering the undesirable outputs.

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


Example #2 `[.ipynb] <https://colab.research.google.com/github/ds2010/pyStoNED/blob/master/sources/notebooks/DDF_UndesirableOutput.ipynb>`_
---------------------------------------------------------------------------------------------------------------------------------------------

Now we take the undesirable outputs into considertion, the python code is presented as 

.. code:: python

    # import packages
    from pystoned import CNLSDDF
    from pystoned.constant import FUN_PROD, OPT_LOCAL
    from pystoned import dataset as dataset
    
    # import the GHG example data
    data = dataset.load_GHG_abatement_cost()
    
    # define and solve the CNLS-DDF model
    model = CNLSDDF.CNLSDDF(y=data.y, x=data.x, b=data.b, fun=FUN_PROD, gx=[0.0, 0.0], gb=-1.0, gy=1.0)
    model.optimize(OPT_LOCAL)

    # display the estimates (alpha, beta, gamma, delta, and residual)
    model.display_alpha()
    model.display_beta()
    model.display_gamma()
    model.display_delta()
    model.display_residual()
    
