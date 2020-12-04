===================================
A ConcreteModel for CNLS estimation
===================================

We also prepare a concrete model that does not need to call any of our developed functions in the pyStoNED package.
In this concrete model, one can define the parameter specification by reading data from an Excel file.

A concreteModel for CNLS estimation `[.ipynb] <https://colab.research.google.com/github/ds2010/pyStoNED/blob/master/docs/advanced/advanced_files/CNLS_ConcreteModel.ipynb>`_ :

.. code:: python
    
    # Import packages
    import pd as pd
    import numpy as np
    import sys
    from pyomo.environ import *

    # Sets
    i = np.array(['i{}'.format(i) for i in range(1, 90)])
    model.i = Set(initialize=i, doc='DMUs', ordered=True )
    model.j = Set(initialize=['j1', 'j2', 'j3'], doc='inputs and outputs')

    # alias
    model.h = SetOf(model.i) 

    # Parameters 
    # output (y1,y2,y3), avaiable at https://github.com/ds2010/pyStoNED/blob/master/sources/data/firms.csv.
    df1 = pd.read_excel('y.xlsx', header=0, index_col=0)
    Dict1 = dict()
    for i in df1.index:
        for j in df1.columns:
            Dict1[i, j] = float(df1[j][i])
    model.y = Param(model.i, model.j, initialize = Dict1) 

    # input (cost)
    df2 = pd.read_excel('x.xlsx', header=0, index_col=0)
    Dict2 = dict()
    for i in df2.index:
            Dict2[i] = float(df2['TOTEX'][i])
    model.c = Param(model.i, initialize = Dict2) 


    #Variables
    model.b = Var(model.i, model.j, bounds=(0.0,None), doc='beta-coeff')
    model.e = Var(model.i, doc='res')
    model.f = Var(model.i, bounds=(0.0,None), doc='frontier')

    # Constraints and objective f.
    def qreg_rule(model, i):
        return log(model.c[i]) == log(model.f[i] + 1) + model.e[i]
    model.qreg = Constraint(model.i, rule=qreg_rule, doc='log-transformed regression')

    def qlog_rule(model, i):
        return model.f[i] == sum(model.b[i, j]*model.y[i, j] for j in model.j) - 1
    model.qlog = Constraint(model.i, rule=qlog_rule, doc='cost function')

    def qconcav_rule(model, i, h):
        return sum(model.b[i,j]*model.y[i,j] for j in model.j) >= sum(model.b[h,j]*model.y[i,j] for j in model.j)
    model.qconcav = Constraint(model.i, model.h, rule=qconcav_rule, doc='concavity constraint')

    def objective_rule(model):
        return sum(model.e[i]*model.e[i] for i in model.i)
    model.objective = Objective(rule=objective_rule, sense=minimize, doc='Define objective function')

    #Solve the model
    solver_manager = SolverManagerFactory('neos')
    results = solver_manager.solve(model, opt='knitro', tee=True)

    #retrive the beta
    ind = list(model.b)
    val = list(model.b[:,:].value)
    df_b = [ i + tuple([j]) for i, j in zip(ind, val)]
    df_b = np.asarray(df_b)
    # display beta coefficients
    df_b

    df_b = pd.DataFrame(df_b,columns = ['Name', 'Key', 'Value'])
    df_b = df_b.pivot(index='Name', columns='Key', values='Value')
    # retrive the residuals
    ind = list(model.e)
    val = list(model.e[:].value)
    df_e = np.asarray(val)
    #display residuals
    df_e