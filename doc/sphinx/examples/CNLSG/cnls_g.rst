======================
Solving CNLS model
======================

Since convex regression approaches shape the convexity (concavity) of function using the Afrait inequality, 
the estimation becomes excessively expensive due to the :math:`O(n^2)` linear constraints. e.g., If the data samples 
have 500 observations, the total number of linear constraints is equal to 250,000. To speed up the computational time, 
Lee et al., (2013) propose a more efficient generic algorithm, CNLS-G, which uses the relaxed Afriat constraint set and 
iteratively adds violated constraints to the relaxed model as necessary. See more discussion in Lee et al., (2013).

To illustrate the CNLS-G algorithm, we follow Lee et al., (2013) to generate the input and output variables. 
In this section, we assume an additive production function with two-input and one-output, :math:`y=x_1^{0.4}*x_2^{0.4}+u`. 
We randomly draw the inputs :math:`x_1` and :math:`x_2` from a uniform distribution, :math:`x \sim U[1, 10]`, and the error term 
:math:`u` from a normal distribution, :math:`u \sim N(0, 0.7^2)`. Based on these specifications, we first generate 
500 artificial observations and then estimate the CNLS problem \eqref{eq:eq2} and the CER problem \eqref{eq:eq8} using the CNLS-G algorithm.

Example `[.ipynb] <https://colab.research.google.com/github/ds2010/pyStoNED/blob/master/notebooks/CNLS_g.ipynb>`_
-------------------------------------------------------------------------------------------------------------------

.. code:: python

    # import packages
    from pystoned import CNLSG, CNLS
    from pystoned.constant import CET_ADDI, FUN_PROD, OPT_LOCAL, RTS_VRS
    import numpy as np
    import time
    
    # set seed
    np.random.seed(0)
    
    # generate DMUs: DGP
    x = np.random.uniform(low=1, high=10, size=(500, 2))
    u = np.random.normal(loc=0, scale=0.7, size=500)
    y = x[:, 0]**0.4*x[:, 1]**0.4+u

    # solve CNLS model without algorithm
    t1 = time.time()
    model1 = CNLS.CNLS(y, x, z=None, cet = CET_ADDI, fun = FUN_PROD, rts = RTS_VRS)
    model1.optimize(OPT_LOCAL)
    CNLS_time = time.time() -t1

    # solve CNLS model using CNLS-G algorithm
    model2 = CNLSG.CNLSG(y, x, z=None, cet = CET_ADDI, fun = FUN_PROD, rts = RTS_VRS)
    model2.optimize(OPT_LOCAL)

    # display running time
    print("The running time with algorithm is ", model2.get_runningtime())
    print("The running time without algorithm is ", CNLS_time)

    # display number of constraints
    print("The total number of constraints is ", model2.get_totalconstr())