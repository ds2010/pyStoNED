====================================
Calculating firm-level efficiency
====================================

Example: Using StoNED with CNLSG `[.ipynb] <https://colab.research.google.com/github/ds2010/pyStoNED/blob/master/notebooks/StoNED_MoM_CNLSG.ipynb>`_
------------------------------------------------------------------------------------------------------------------------------------
    
.. code:: python
    
    # import packages
    from pystoned import CNLSG, StoNED
    from pystoned.dataset import load_Finnish_electricity_firm
    from pystoned.constant import CET_MULT, FUN_COST, RTS_VRS, RED_MOM
    
    # import Finnish electricity distribution firms data
    data = load_Finnish_electricity_firm(x_select=['OPEX', 'CAPEX'], y_select=['Energy'])
    
    # build and optimize the CNLS model
    model = CNLSG.CNLSG(data.y, data.x, z=None, cet=CET_MULT, fun=FUN_COST, rts=RTS_VRS)
    model.optimize('email@address')
    
    # Calculate firm-level efficiency
    rd = StoNED.StoNED(model)
    print(rd.get_technical_inefficiency(RED_MOM))
    