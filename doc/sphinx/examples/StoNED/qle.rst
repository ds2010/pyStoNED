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


Example `[.ipynb] <https://colab.research.google.com/github/ds2010/pyStoNED/blob/master/notebooks/StoNED_QLE.ipynb>`_
------------------------------------------------------------------------------------------------------------------------

In the following code, we use the quassi-likelihood approach to decompose the CNLS residuals and display the StoNED frontier.

.. code:: python

    # import packages
    from pystoned import CNLS, StoNED
    from pystoned.dataset import load_Finnish_electricity_firm
    from pystoned.constant import CET_MULT, FUN_COST, RTS_VRS, RED_QLE
    
    # import Finnish electricity distribution firms data
    data = load_Finnish_electricity_firm(x_select=['OPEX', 'CAPEX'], y_select=['Energy'])
    
    # build and optimize the CNLS model
    model = CNLS.CNLS(data.y, data.x, z=None, cet=CET_MULT, fun=FUN_COST, rts=RTS_VRS)
    model.optimize('email@address')
    
    # Residual decomposition
    rd = StoNED.StoNED(model)
    print(rd.get_technical_inefficiency(RED_QLE))
    
    # return the StoNED frontier
    print(rd.get_frontier(RED_QLE))
