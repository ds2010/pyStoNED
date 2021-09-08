Quassi-likelihood estimation
=============================

Quassi-likelihood approach is alternative to decomposing the :math:`\sigma_u` and :math:`\sigma_v` suggested
by Fan et al. (1996). In this method we apply the standard maximum likelihood method to 
estimate the parameters :math:`\sigma_u` and :math:`\sigma_v`, taking the shape of ``CNLS`` curve
as given. The quasi-likelihood function is formulated as

.. math::
    :nowrap:

    \begin{align*}
        \text{ln} L(\lambda) & = -n\text{ln}(\hat{\sigma}) + \sum \text{ln} \Phi\bigg[\frac{-\hat{\varepsilon}_i \lambda}{\hat{\sigma}}\bigg] - \frac{1}{2\hat{\sigma}^2}\sum\hat{\varepsilon}_i^2 
    \end{align*}

where

.. math::
    :nowrap:
    
    \begin{align*}
        \hat{\varepsilon}_i &= \hat{\varepsilon}_i^{CNLS}-(\sqrt{2}\lambda\hat{\sigma})/[\pi(1+\lambda^2)]^{1/2}    \\
        \hat{\sigma} &= \Bigg\{\frac{1}{n}\sum(\hat{\varepsilon}_i^{CNLS})^2 / \bigg[1 - \frac{2\lambda^2}{\pi(1+\lambda^2)}\bigg]  \Bigg\}  
    \end{align*}

Note that the quasi-likelihood function only consists of a single parameter :math:`\lambda` (i.e., the signal-to-noise ratio :math:`\lambda = \sigma_u/\sigma_v`).  
The symbol :math:`\Phi` represents the cumulative distribution function of the standard normal distribution. In the `pyStoNED` 
package, we use the Broyden–Fletcher–Goldfarb–Shanno (BFGS) algorithm to solve the maximum likelihood function.


Example: StoNED with CNLS `[.ipynb] <https://colab.research.google.com/github/ds2010/pyStoNED/blob/master/notebooks/StoNED_QLE_CNLS.ipynb>`_
-----------------------------------------------------------------------------------------------------------------------------------------------

In the following code, we use the quassi-likelihood approach to decompose the CNLS residuals and display the StoNED frontier.

.. code:: python

    # import packages
    from pystoned import CNLS, StoNED
    from pystoned.dataset import load_Finnish_electricity_firm
    from pystoned.constant import CET_MULT, FUN_COST, RTS_VRS, RED_QLE
    
    # import Finnish electricity distribution firms data
    data = load_Finnish_electricity_firm(x_select=['Energy', 'Length', 'Customers'],
                                        y_select=['TOTEX'])
    
    # build and optimize the CNLS model
    model = CNLS.CNLS(data.y, data.x, z=None, cet=CET_MULT, fun=FUN_COST, rts=RTS_VRS)
    model.optimize('email@address')
    
    # calculate and print unconditional expected inefficiency (mu)
    rd = StoNED.StoNED(model)
    print(rd.get_unconditional_expected_inefficiency(RED_QLE))
    
