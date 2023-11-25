Estimating the StoNED frontier
===============================

Having estimated the expected inefficiency :math:`\hat{\mu}`, we can shift the conditional mean function estimated by CNLS to the frontier in the case of the additive and multiplicative models under VRS and CRS. Given the fact that the function estimated by CNLS is only unique for the observed points :math:`\bm{x}_i`, we then follow Kuosmanen and Kortelainen (2012) to estimate the unique conditional mean function :math:`\hat{g}_{min}^{CNLS}(\bm{x})` under VRS. 

.. math::
    :nowrap:

    \begin{equation*}
        \hat{g}_{min}^{CNLS}(\bm{x}) = \underset{\alpha, \boldsymbol{\beta}}{\min}\{\alpha + \boldsymbol{\beta}^{\prime}x \mid \alpha + \boldsymbol{\beta}^{\prime}\bm{x}_i \ge \hat{g}^{CNLS}(\bm{x}_i), \forall i\}
    \end{equation*}

where :math:`\hat{g}^{CNLS}(\bm{x}_i)`` is the conditional mean function estimated by the CNLS estimator. We subsequently shift the conditional mean function :math:`\hat{g}_{min}^{CNLS}(\bm{x})` to the frontier using 

- Production function.
    
    1. Additive model: :math:`\hat{f}^{StoNED}(\bm{x}) = \hat{g}_{min}^{CNLS}(\bm{x}) + \hat{\mu}`;
    2. Multiplicative model: :math:`\hat{f}^{StoNED}(\bm{x}) = \hat{g}_{min}^{CNLS}(\bm{x}) \cdot \exp(\hat{\mu})`.

- Cost function.

    1. Additive model: :math:`\hat{f}^{StoNED}(\bm{x}) = \hat{g}_{min}^{CNLS}(\bm{x}) - \hat{\mu}`;
    2. Multiplicative model: :math:`\hat{f}^{StoNED}(\bm{x}) = \hat{g}_{min}^{CNLS}(\bm{x}) \cdot \exp(-\hat{\mu})`.


Example: StoNED frontier `[.ipynb] <https://colab.research.google.com/github/ds2010/pyStoNED/blob/master/notebooks/StoNED_frontier.ipynb>`_
-----------------------------------------------------------------------------------------------------------------------------------------------

In the following example, we demonstrate how to obtain the StoNED frontier via the package.

.. code:: python

    # import packages
    from pystoned import CNLS, StoNED
    from pystoned.dataset import load_Finnish_electricity_firm
    from pystoned.constant import CET_MULT, FUN_COST, RTS_VRS, RED_MOM
    
    # import Finnish electricity distribution firms data
    data = load_Finnish_electricity_firm(x_select=['Energy', 'Length', 'Customers'],
                                        y_select=['TOTEX'])
    
    # build and optimize the CNLS model
    model = CNLS.CNLS(data.y, data.x, z=None, cet=CET_MULT, fun=FUN_COST, rts=RTS_VRS)
    model.optimize('email@address')
    
    # calculate and print the StoNED frontier
    rd = StoNED.StoNED(model)
    print(rd.get_stoned(RED_MOM))
    