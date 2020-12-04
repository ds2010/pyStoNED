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



Example #1 `[.ipynb] <https://colab.research.google.com/github/ds2010/pyStoNED/blob/master/sources/notebooks/DDF_withoutUndesirableOutput.ipynb>`_
-----------------------------------------------------------------------------------------------------------------------------------------------------

In the following code, we first estimate the DDF without considering the undesirable outputs.

.. code:: python

    # import packages
    from pystoned import CNLSDDF
    import pandas as pd
    import numpy as np
    
    # import Finnish electricity distribution firms data
    url='https://raw.githubusercontent.com/ds2010/pyStoNED/master/sources/data/firms.csv'
    df = pd.read_csv(url, error_bad_lines=False)
    df.head(5)
    
    # outputs
    y1 = df['Energy']
    y1 = np.asmatrix(y1).T
    y2 = df['Length']
    y2 = np.asmatrix(y2).T
    y3 = df['Customers']
    y3 = np.asmatrix(y3).T
    y = np.concatenate((y1, y2, y3), axis=1)

    # inputs
    x1 = df['OPEX']
    x1 = np.asmatrix(x1).T
    x2 = df['CAPEX']
    x2 = np.asmatrix(x2).T
    x = np.concatenate((x1, x2), axis=1)

    # define and solve the CNLS-DDF model
    model = CNLSDDF.CNLSDDF(y, x, b=None, fun = "prod", gx= [0.0, 0.0], gb=None, gy= [0.0, 0.5, 0.5])
    model.optimize(remote=False)

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
    import pandas as pd
    import numpy as np
    
    # import OECD countries data
    url = 'https://raw.githubusercontent.com/ds2010/pyStoNED/master/sources/data/countries.csv'
    df = pd.read_csv(url, error_bad_lines=False)
    df.head(5)
    
    # inputs
    x1 = df['HRSN']
    x1 = np.asmatrix(x1).T
    x2 = df['CPNK']
    x2 = np.asmatrix(x2).T
    x = np.concatenate((x1, x2), axis=1)

    # good output
    y = df['VALK']

    # bad output
    b = df['GHG']

    # define and solve the CNLS-DDF model
    model = CNLSDDF.CNLSDDF(y, x, b, fun="prod", gx=[0.0, 0.0], gb=-1.0, gy=1.0)
    model.optimize(remote=False)

    # display the estimates (alpha, beta, gamma, delta, and residual)
    model.display_alpha()
    model.display_beta()
    model.display_gamma()
    model.display_delta()
    model.display_residual()
    