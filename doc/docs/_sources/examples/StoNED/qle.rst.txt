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


Example `[.ipynb] <https://colab.research.google.com/github/ds2010/pyStoNED/blob/master/sources/notebooks/StoNED_QLE.ipynb>`_
-------------------------------------------------------------------------------------------------------------------------------

In the following code, we use the quassi-likelihood approach to decompose the CNLS residuals with the pyStoNED.

.. code:: python

    # import packages
    from pystoned import StoNED
    import pandas as pd
    import numpy as np
    
    # import Finnish electricity distribution firms data
    url='https://raw.githubusercontent.com/ds2010/pyStoNED/master/sources/data/firms.csv'
    df = pd.read_csv(url, error_bad_lines=False)
    df.head(5)
    
    # output
    y = df['Energy']

    # inputs
    x1 = df['OPEX']
    x1 = np.asmatrix(x1).T
    x2 = df['CAPEX']
    x2 = np.asmatrix(x2).T
    x = np.concatenate((x1, x2), axis=1)

    # define and solve the StoNED model using QLE approach
    model = StoNED.StoNED(y, x, z= None, cet = "mult", fun = "cost", rts = "vrs")
    model.optimize(remote=True)

    # retrive the unconditional expected inefficiency \mu
    print(model.get_unconditional_expected_inefficiency('QLE'))

    # retrive the technical inefficiency
    print(model.get_technical_inefficiency(method='QLE'))
