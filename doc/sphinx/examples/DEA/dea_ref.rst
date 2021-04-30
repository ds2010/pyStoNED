================================
DEA: Change reference set
================================


Sometimes we need to draw a distinction between the set of **reference** DMUs that characterize the 
frontier and the set of **evaluated** DMUs used as in the DEA problem. 
However, by default, all DMUs serve both as benchmarks and evaluated units.

Example `[.ipynb] <https://colab.research.google.com/github/ds2010/pyStoNED/blob/master/notebooks/notebooks/DEA_changeReferenceSet.ipynb.ipynb>`_
---------------------------------------------------------------------------------------------------------------------------------------------------

In the following code, we calculate the VRS radial model with pyStoNED.

.. code:: python

    # import packages
    from pystoned import DEA
    from pystoned.constant import RTS_VRS, ORIENT_OO, OPT_LOCAL
    
    # evaluated DMUs
    x = np.array([100,200,300,500,100,200,600,400,550,600])
    y = np.array([75,100,300,400,25,50,400, 260, 180, 240])

    # reference DMUs
    xref = np.array([100,300,500,100,600])
    yref = np.array([75,300,400,25,400])

    # define and solve the DEA radial model
    model = DEA.DEA(y, x, rts=RTS_VRS, orient=ORIENT_OO, yref=yref, xref=xref)
    model.optimize(OPT_LOCAL)

    # display the technical efficiency
    model.display_theta()

    # display the intensity variables
    model.display_lamda()
