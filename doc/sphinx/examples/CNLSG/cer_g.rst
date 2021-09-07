======================
Solving CER model
======================

Example: Solving CER `[.ipynb] <https://colab.research.google.com/github/ds2010/pyStoNED/blob/master/notebooks/CQR_g.ipynb>`_
--------------------------------------------------------------------------------------------------------------------------------

.. code:: python

    # import packages
    from pystoned import CQERG, CQER
    from pystoned.constant import CET_ADDI, FUN_PROD, OPT_LOCAL, RTS_VRS
    import numpy as np
    import time
    
    # set seed
    np.random.seed(0)
    
    # generate DMUs: DGP
    x = np.random.uniform(low=1, high=10, size=(500, 2))
    u = np.random.normal(loc=0, scale=0.7, size=500)
    y = x[:, 0]**0.4*x[:, 1]**0.4+u

    # solve CER model without algorithm
    tau = 0.9
    t1 = time.time()
    model1 = CQER.CER(y, x, tau, z=None, cet = CET_ADDI, fun = FUN_PROD, rts = RTS_VRS)
    model1.optimize(OPT_LOCAL)
    CER_time = time.time() -t1
    
    # solve CER model using CNLS-G algorithm
    model2 = CQERG.CERG(y, x, tau, z=None, cet = CET_ADDI, fun = FUN_PROD, rts = RTS_VRS)
    model2.optimize(OPT_LOCAL)

    # display running time
    print("The running time with algorithm is ", model2.get_runningtime())
    print("The running time without algorithm is ", CER_time)

    # display number of constraints
    print("The total number of constraints in CER model is ", model2.get_totalconstr())