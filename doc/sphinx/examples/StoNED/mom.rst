Method of Moments
===================

The method of moments requires some additional parametric distributional assumptions. 
Following Kuosmanen et al. (2015), under the maintained assumptions of half-normal inefficiency and normal noise, 
the second and third central moments of the composite error distribution are given 
by

.. math::
    :nowrap:

    \begin{align*}
        M_2 &= \bigg[\frac{\pi-2}{\pi}\bigg] \sigma_u^2 + \sigma_v^2  \\
        M_3 &= \bigg(\sqrt{\frac{2}{\pi}}\bigg)\bigg[1-\frac{4}{\pi}\bigg]\sigma_u^2
    \end{align*}

These central moments can be estimated by using the CNLS residuals:

.. math::
    :nowrap:
    
    \begin{align*}
    \hat{M_2} &= \sum_{i=1}^{n}(\hat{\varepsilon}_i-\bar{\varepsilon})^{2}/n  \\
    \hat{M_3} &= \sum_{i=1}^{n}(\hat{\varepsilon}_i-\bar{\varepsilon})^{3}/n  
    \end{align*}

Note that the third moment :math:`M_3` (which measures the skewness of the distribution) 
only depends on the standard deviation parameter σu of the inefficiency distribution. 
Thus, given the estimated :math:`\hat{M}_3` (which should be positive in the case of a cost 
frontier), we can estimate :math:`\sigma_u` parameter by

.. math::
    :nowrap:
    
    \begin{align*}
        \hat{\sigma}_u &= \sqrt[3]{\frac{\hat{M_3}}{\bigg(\sqrt{\frac{2}{\pi}}\bigg)\bigg[1-\frac{4}{\pi}\bigg]}} \\
        \hat{\sigma}_v &= \sqrt[2]{\hat{M_2}-\bigg[\frac{\pi-2}{\pi}\bigg] \hat{\sigma}_u^2 }
    \end{align*}


Example `[.ipynb] <https://colab.research.google.com/github/ds2010/pyStoNED/blob/master/notebooks/StoNED_MoM.ipynb>`_
-------------------------------------------------------------------------------------------------------------------------------

In the following code, we use the method of moments to decompose the CNLS residuals and display the StoNED frontier.

.. code:: python

    # import packages
    from pystoned import CNLS, StoNED
    from pystoned.dataset import load_Finnish_electricity_firm
    from pystoned.constant import CET_MULT, FUN_COST, RTS_VRS, RED_MOM
    
    # import Finnish electricity distribution firms data
    data = load_Finnish_electricity_firm(x_select=['OPEX', 'CAPEX'], y_select=['Energy'])
    
    # build and optimize the CNLS model
    model = CNLS.CNLS(data.y, data.x, z=None, cet=CET_MULT, fun=FUN_COST, rts=RTS_VRS)
    model.optimize('email@address')
    
    # Residual decomposition
    rd = StoNED.StoNED(model)
    print(rd.get_technical_inefficiency(RED_MOM))
    
    # return the StoNED frontier
    print(rd.get_frontier(RED_MOM))