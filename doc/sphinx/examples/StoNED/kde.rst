Kernel deconvolution estimation
=================================

While both method of moments and quasi-likelihood techniques require parametric assumptions,
a fully nonparametric alternative is available for estimating the signal-to-noise ratio :math:`\lambda`,
as proposed by Hall and Simar (2002). A robust nonparametric estimator of expected inefficiency
:math:`\mu` is obtained as

.. math::
    :nowrap:

    \begin{align*}
        \hat{\mu} = \text{arg} \max_{z \in C}(\hat{f^{'}}_{\varepsilon^+}(Z)),
    \end{align*}

where :math:`C` is a closed interval in he right tail of :math:`f_{\varepsilon^+}`.


Example `[.ipynb] <https://colab.research.google.com/github/ds2010/pyStoNED/blob/master/notebooks/StoNED_KDE.ipynb>`_
-------------------------------------------------------------------------------------------------------------------------------

In the following code, we use the kernel density approach to decompose the CNLS residuals and display the unconditional expected inefficiency.

.. code:: python

    # import packages
    from pystoned import CNLS, StoNED
    from pystoned.dataset import load_Finnish_electricity_firm
    from pystoned.constant import CET_MULT, FUN_COST, RTS_VRS, RED_KDE
    
    # import Finnish electricity distribution firms data
    data = load_Finnish_electricity_firm(x_select=['OPEX', 'CAPEX'], y_select=['Energy'])
    
    # build and optimize the CNLS model
    model = CNLS.CNLS(data.y, data.x, z=None, cet=CET_MULT, fun=FUN_COST, rts=RTS_VRS)
    model.optimize('email@address')
    
    # retrive the unconditional expected inefficiency \mu
    model = StoNED.StoNED(model)
    print(model.get_unconditional_expected_inefficiency('KDE'))
