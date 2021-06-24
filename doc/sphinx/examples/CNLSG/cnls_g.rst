===================
CNLS-G Alogrithm
===================


Since convex regression approaches shape the convexity (concavity) of function using the Afrait inequality, 
the estimation becomes excessively expensive due to the :math:`O(n^2)` linear constraints. e.g., If the sample has 
500 observations, the total linear constraints is 250,000. To speed up the computational time, Lee et al., (2013) 
propose a more efficient generic algorithm, CNLS-G, which uses the relaxed Afriat constraint set and iteratively 
adds violated constraints to the relaxed model as necessary. See more discussion in Lee et al., (2013).

Example: CNLS model `[.ipynb] <https://colab.research.google.com/github/ds2010/pyStoNED/blob/master/notebooks/CNLS_g.ipynb>`_
--------------------------------------------------------------------------------------------------------------------------------

.. code:: python

    # import packages
    from pystoned import CNLSG
    from pystoned.constant import CET_ADDI, FUN_PROD, OPT_LOCAL, RTS_VRS
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
    model = CNLSG.CNLSG(y, x, z=None, cet = CET_ADDI, fun = FUN_PROD, rts = RTS_VRS)
    model.optimize(OPT_LOCAL)

    # display running time
    print("The running time is " model.get_runningtime())

    # display number of constraints
    print("The total number of constraints is " model.get_totalconstr())


Example: CQR/CER model `[.ipynb] <https://colab.research.google.com/github/ds2010/pyStoNED/blob/master/notebooks/CQR_g.ipynb>`_
----------------------------------------------------------------------------------------------------------------------------------

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
    model1 = CQERG.CQRG(y, x, tau, z=None, cet = CET_ADDI, fun = FUN_PROD, rts = RTS_VRS)
    model1.optimize(OPT_LOCAL)
    
    # CER model
    model2 = CQERG.CERG(y, x, tau, z=None, cet = CET_ADDI, fun = FUN_PROD, rts = RTS_VRS)
    model2.optimize(OPT_LOCAL)

    # display running time
    print("The running time for CQR estimation is " model1.get_runningtime())
    print("The running time for CER estimation is " model2.get_runningtime())

    # display number of constraints
    print("The total number of constraints in CQR model is " model1.get_totalconstr())
    print("The total number of constraints in CER model is " model2.get_totalconstr())
