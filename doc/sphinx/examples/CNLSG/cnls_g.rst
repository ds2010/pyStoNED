===========================================
CNLS-G Alogrithm: CNLS and CQR/CER models
===========================================


Since both CNLS and CQR/CER models have the Afrait inequality constraint, the estimation becomes
excessively expensive due to the :math:`O(n^2)` linear constraints. e.g., If the sample
has 500 observations, the total linear constraints is 250,000. Lee et al., (2013) thus proposed
a more efficient algorithm, CNLS-G, to improve the computational efficiency. See more discussion in Lee et al. (2013).

Here, we demostrate how to use the CNLS-G algorithm to solve the CNLS and CQR models: 

Example: CNLS model `[.ipynb] <https://colab.research.google.com/github/ds2010/pyStoNED/blob/master/notebooks/CNLS_g.ipynb>`_
-----------------------------------------------------------------------------------------------------------------------------------

.. code:: python

    # import packages
    from pystoned import CNLSG
    from pystoned.constant import CET_ADDI, FUN_COST, OPT_LOCAL, RTS_VRS
    import numpy as np
    
    # set seed
    np.random.seed(0)
    
    # generate DMUs: DGP
    x1 = np.random.uniform(low=1, high=10, size=500)
    x2 = np.random.uniform(low=1, high=10, size=500)
    u = np.random.normal(loc=0, scale=0.7, size=500)
    y = x1**0.4*x2**0.4+u
    x = np.concatenate((x1.reshape(500, 1), x2.reshape(500, 1)), axis=1)

    # define and solve the CNLS model
    model = CNLSG.CNLSG(y, x, z=None, cet = CET_ADDI, fun = FUN_COST, rts = RTS_VRS)
    model.optimize(OPT_LOCAL)

    # display running time
    print(model.get_runningtime())

    # display residuals
    model.display_residual()



Example: CQR/CER model `[.ipynb] <https://colab.research.google.com/github/ds2010/pyStoNED/blob/master/notebooks/CQR_g.ipynb>`_
------------------------------------------------------------------------------------------------------------------------------------

.. code:: python

    # import packages
    from pystoned import CQERG
    from pystoned.constant import CET_ADDI, FUN_COST, OPT_LOCAL, RTS_VRS
    import numpy as np
    
    # set seed
    np.random.seed(0)
    
    # generate DMUs: DGP
    x1 = np.random.uniform(low=1, high=10, size=500)
    x2 = np.random.uniform(low=1, high=10, size=500)
    u = np.random.normal(loc=0, scale=0.7, size=500)
    y = x1**0.4*x2**0.4+u
    x = np.concatenate((x1.reshape(500, 1), x2.reshape(500, 1)), axis=1)

    # define and solve the CQR/CER model
    tau = 0.9
    # CQR model
    model = CQERG.CQRG(y, x, tau, z=None, cet = CET_ADDI, fun = FUN_COST, rts = RTS_VRS)
    # OR CER model
    # model = CQERG.CERG(y, x, tau, z=None, cet = CET_ADDI, fun = FUN_COST, rts = RTS_VRS)
    model.optimize(OPT_LOCAL)

    # display running time
    print(model.get_runningtime())

    # display residuals
    model.display_residual()
