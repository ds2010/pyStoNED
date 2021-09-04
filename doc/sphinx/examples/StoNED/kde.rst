Kernel deconvolution estimation
=================================

while method of moments and quasi-likelihood approaches require additional distributional assumptions
for the inefficiency and noise terms, a fully nonparametric estimation of the expected inefficiency
:math:`\mu` is also available by applying nonparametric kernel deconvolution, proposed by Hall and Simar (2002). 
Note that the residuals :math:`\hat{\varepsilon}_i^{CNLS}` are consistent estimators of :math:`e^o = \varepsilon_i + \mu` (for production model).
Following Kuosmanen and Johnson (2017), the density function of :math:`{e^o}`

.. math::
    :nowrap:
    
    \begin{align*}
        \hat{f}_{e^o}(z) = (nh)^{-1} \sum_{i=1}^{n}K\bigg(\frac{z-e_i^o}{h} \bigg)
    \end{align*}

where :math:`K(\cdot)` is a compactly supported kernel and :math:`h` is a bandwidth. 
\citet{Hall2002} show that the first derivative of the density function of 
the composite error term (:math:`f_\varepsilon^{'}`)is proportional to that of the 
inefficiency term (:math:`f_u^{'}`) in the neighborhood of :math:`\mu`. Therefore, 
a robust nonparametric estimator of expected inefficiency :math:`\mu` is obtained as

.. math::
    :nowrap:

    \begin{align*}
        \hat{\mu} = \arg \max_{z \in C}(\hat{f^{'}}_{e^o}(z)),
    \end{align*}

where :math:`C` is a closed interval in he right tail of :math:`f_{e^o}`.


Example: StoNED with CNLS `[.ipynb] <https://colab.research.google.com/github/ds2010/pyStoNED/blob/master/notebooks/StoNED_KDE.ipynb>`__
------------------------------------------------------------------------------------------------------------------------

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
    
    # calculate and print unconditional expected inefficiency (mu)
    rd = StoNED.StoNED(model)
    print(rd.get_unconditional_expected_inefficiency(RED_KDE))
