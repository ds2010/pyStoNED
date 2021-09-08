Method of moments
===================

The method of moments requires some additional parametric distributional assumptions. 
Following Kuosmanen et al. (2015), under the maintained assumptions of half-normal inefficiency 
(i.e., :math:`u_i \sim N^+(0, \sigma_u^2)`) and normal noise (i.e., :math:`v_i \sim N(0, \sigma_v^2)`), 
the second and third central moments of the composite error (i.e., :math:`\varepsilon_i`) 
distribution are given by

.. math::
    :nowrap:

    \begin{align*}
        M_2 &= \bigg[\frac{\pi-2}{\pi}\bigg] \sigma_u^2 + \sigma_v^2  \\
        M_3 &= \bigg(\sqrt{\frac{2}{\pi}}\bigg)\bigg[1-\frac{4}{\pi}\bigg]\sigma_u^2
    \end{align*}

    The second and third central moments can be estimated by using the CNLS residuals, i.e.,  :math:`\hat{\varepsilon}_i^{CNLS}`:

.. math::
    :nowrap:
    
    \begin{align*}
    \hat{M_2} &= \sum_{i=1}^{n}(\hat{\varepsilon}_i-\bar{\varepsilon})^{2}/n  \\
    \hat{M_3} &= \sum_{i=1}^{n}(\hat{\varepsilon}_i-\bar{\varepsilon})^{3}/n  
    \end{align*}

Note that the third moment :math:`M_3` (which measures the skewness of the distribution) 
only depends on the standard deviation parameter σu of the inefficiency distribution. 
Thus, given the estimated :math:`\hat{M}_3` (which should be positive in the case of a cost 
frontier), we can estimate :math:`\sigma_u` and :math:`\sigma_v` by

.. math::
    :nowrap:
    
    \begin{align*}
        \hat{\sigma}_u &= \sqrt[3]{\frac{\hat{M_3}}{\bigg(\sqrt{\frac{2}{\pi}}\bigg)\bigg[1-\frac{4}{\pi}\bigg]}} \\
        \hat{\sigma}_v &= \sqrt[2]{\hat{M_2}-\bigg[\frac{\pi-2}{\pi}\bigg] \hat{\sigma}_u^2 }
    \end{align*}


Example: StoNED with CNLS `[.ipynb] <https://colab.research.google.com/github/ds2010/pyStoNED/blob/master/notebooks/StoNED_MoM_CNLS.ipynb>`_
-----------------------------------------------------------------------------------------------------------------------------------------------


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
    
    # calculate and print unconditional expected inefficiency (mu)
    rd = StoNED.StoNED(model)
    print(rd.get_unconditional_expected_inefficiency(RED_MOM))

