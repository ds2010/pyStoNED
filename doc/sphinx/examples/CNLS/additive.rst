============================
Additive CNLS model 
============================

Hildreth (1954) is the first to consider the nonparametric regression subject to monotonicity and concavity constraints 
in the case of a single input variable :math:`x`. Kuosmanen (2008) extends Hildreth’s approach to the multivariate setting 
with the multidimensional input :math:`\boldsymbol{x}`, and refers it to as the Convex Nonparametric Least Squares (CNLS). CNLS builds 
upon the assumption that the true but unknown production function :math:`f` belongs to the set of continuous, monotonic increasing 
and globally concave (convex) functions, imposing exactly the same production axioms as standard Data Envelopment Analysis (DEA) 
(see further discussion in Kuosmanen and Johnson, 2010). The additive multivariate CNLS formulations are defined as

- Esimating production function

.. math::
    :nowrap:
    
    \begin{alignat}{2}
        \underset{\alpha, \boldsymbol{\beta}, \varepsilon} \min &\sum_{i=1}^n\varepsilon_i^2  \\
        \textit{s.t.}\quad 
        &  y_i = \alpha_i + \boldsymbol{\beta}_i^{'}\boldsymbol{x}_i + \varepsilon_i &{\quad}& \forall i \notag \\
        &  \alpha_i + \boldsymbol{\beta}_i^{'}\boldsymbol{x}_i \le \alpha_j + \boldsymbol{\beta}_j^{'}\boldsymbol{x}_i  &{\quad}& \forall i, j \notag\\
        &  \boldsymbol{\beta}_i \ge \boldsymbol{0} &{\quad}& \forall i \notag
    \end{alignat}

- Estimating cost function

.. math::
    :nowrap:
    
    \begin{alignat}{2}
        \underset{\alpha, \boldsymbol{\beta}, \varepsilon} \min & \sum_{i=1}^n\varepsilon_i^2 \\
        \textit{s.t.}\quad 
        &  y_i = \alpha_i + \boldsymbol{\beta}_i^{'}\boldsymbol{x}_i + \varepsilon_i &{\quad}& \forall i  \notag \\
        &  \alpha_i + \boldsymbol{\beta}_i^{'}\boldsymbol{x}_i \ge \alpha_j + \boldsymbol{\beta}_j^{'}\boldsymbol{x}_i  &{\quad}&  \forall i, j  \notag \\
        &  \boldsymbol{\beta}_i \ge \boldsymbol{0}  &{\quad}& \forall i \notag
    \end{alignat}

where :math:`\alpha_i` and :math:`\boldsymbol{\beta}_i` define the intercept and slope parameters of tangent 
hyperplanes that characterize the estimated piecewise linear frontier. :math:`\varepsilon_i`
denotes the CNLS residuals. The first constraint can be interpreted as a multivariate 
regression equation, the second constraint imposes convexity (concavity), and the third 
constraint imposes monotonicity. Similar to the DEA specification, other standard specification 
of returns to scale can be imposed by an additional constraint on the intercept term :math:`\alpha_i`. 
If :math:`\alpha_i=0`, then the problem (2.2) or (2.2) is a constant returns to scale (CRS) 
model, otherwise it is a variable returns to scale (VRS) model. 

The additive CNLS model can be estimated in Python using the module CNLS(y, x, z, ...)
in the package ``pyStoNED`` with the cet parameter set to CET\_ADDI (additive model), 
rts parameter set to RTS\_VRS (VRS model), and \code{fun} parameter set to FUN\_PROD
(production function) or FUN\_COST (cost function). Further, in this section we set the parameter 
z=None and introduce it in section (2.2). The results can be displayed in the screen directly 
using the .display\_alpha() (i.e., display the coefficients :math:`\hat{\alpha}_i`) or stored in the memory 
using the .get\_alpha().

The following examples are to demonstrate how to estimate the VRS models:

Example: production model `[.ipynb] <https://colab.research.google.com/github/ds2010/pyStoNED/blob/master/notebooks/CNLS_prod.ipynb>`_
--------------------------------------------------------------------------------------------------------------------------------------------

.. code:: python

    # import packages
    from pystoned import CNLS
    from pystoned.constant import CET_ADDI, FUN_PROD, OPT_LOCAL, RTS_VRS
    from pystoned.dataset import load_Finnish_electricity_firm
    
    # import Finnish electricity distribution firms data
    data = load_Finnish_electricity_firm(x_select=['Energy', 'Length', 'Customers'],
                                          y_select=['TOTEX'])

    # define and solve the CNLS model
    model = CNLS.CNLS(y=data.y, x=data.x, z=None, 
                        cet = CET_ADDI, fun = FUN_PROD, rts = RTS_VRS)
    
    # estimate the model: 1) local estimation; 2) remote estimation
    model.optimize(OPT_LOCAL)

    # Please replace with your own email address reqired by NEOS server 
    #               (see https://neos-guide.org/content/FAQ#email)
    # model.optimize('email@address') 

    # display the estimates
    model.display_alpha()
    model.display_beta()
    model.display_residual()

    # store the estimates
    alpha = model.get_alpha()
    beta = model.get_beta()
    residuals = model.get_residual()


Example: cost model `[.ipynb] <https://colab.research.google.com/github/ds2010/pyStoNED/blob/master/notebooks/CNLS_cost.ipynb>`_
----------------------------------------------------------------------------------------------------------------------------------------

.. code:: python

    # import packages
    from pystoned import CNLS
    from pystoned.constant import CET_ADDI, FUN_COST, OPT_LOCAL, RTS_VRS
    from pystoned.dataset import load_Finnish_electricity_firm
    
    # import Finnish electricity distribution firms data
    data = load_Finnish_electricity_firm(x_select=['Energy', 'Length', 'Customers'],
                                            y_select=['TOTEX'])
    
    # define and solve the CNLS model
    model = CNLS.CNLS(y=data.y, x=data.x, z=None, 
                        cet = CET_ADDI, fun = FUN_COST, rts = RTS_VRS)
    model.optimize(OPT_LOCAL)

    # display residuals
    model.display_residual()