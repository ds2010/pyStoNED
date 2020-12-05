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

In the following code, we use the method of moments to decompose the CNLS residuals with the pyStoNED.

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

    # define and solve the StoNED model using MoM approach
    model = StoNED.StoNED(y, x, z= None, cet = "mult", fun = "cost", rts = "vrs")
    model.optimize(remote=True)

    # retrive the unconditional expected inefficiency \mu
    print(model.get_unconditional_expected_inefficiency('MOM'))

    # retrive the technical inefficiency
    print(model.get_technical_inefficiency(method='MOM'))
