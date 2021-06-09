================================================================
Plot of estimated function/frontier: two-input and one-output
================================================================


Example `[.ipynb] <https://colab.research.google.com/github/ds2010/pyStoNED/blob/master/notebooks/3dplot.ipynb>`_
--------------------------------------------------------------------------------------------------------------------

.. code:: python

    # import packages
    from pystoned import CNLS
    from pystoned.plot import plot3d
    from pystoned.constant import CET_ADDI, FUN_PROD, OPT_LOCAL, RTS_VRS
    from pystoned.dataset import load_Finnish_electricity_firm

    # import Finnish electricity distribution firms data
    data = load_Finnish_electricity_firm(x_select=['Energy', 'Length'],
                                            y_select=['TOTEX'])

    # define and solve the CNLS model
    model = CNLS.CNLS(y=data.y, x=data.x, z=None, cet = CET_ADDI, fun = FUN_PROD, rts = RTS_VRS)
    model.optimize(OPT_LOCAL)

    # Plot the estimated function
    plot3d(model,x_select_1=0, x_select_2=1, fig_name="CNLS_3d")

.. image:: ../../../../notebooks/CNLS_3d.png
    :width: 600
    :alt: CNLS estimated function