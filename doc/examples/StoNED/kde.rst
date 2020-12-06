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

In the following code, we use the kernel density approach to decompose the CNLS residuals with the pyStoNED.

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

    # define and solve the StoNED model using KDE approach
    model = StoNED.StoNED(y, x, z= None, cet = "mult", fun = "cost", rts = "vrs")
    model.optimize(remote=True)

    # retrive the unconditional expected inefficiency \mu
    print(model.get_unconditional_expected_inefficiency('KDE')) 
