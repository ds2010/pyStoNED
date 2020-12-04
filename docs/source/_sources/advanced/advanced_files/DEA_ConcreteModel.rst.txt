===================================
A ConcreteModel for DEA estimation
===================================

We also prepare a concrete model that 
can be used to calculate the input oriented VRS model.

A ConcreteModel for DEA estimation `[.ipynb] <https://colab.research.google.com/github/ds2010/pyStoNED/blob/master/docs/advanced/advanced_files/DEA_ConcreteModel.ipynb>`_ :

.. code:: python
    
    # import PYOMO package
    from pyomo.environ import *
    # creat a concrete model
    model = ConcreteModel()

    # Sets
    model.i = Set(initialize=['i1', 'i2','i3', 'i4','i5', 'i6','i7', 'i8','i9', 'i10'], doc='DMUs', ordered=True)
    model.j = Set(initialize=['OPEX', 'CAPEX'])   # inputs
    model.k = Set(initialize=['Energy', 'Length', 'Customers'])   #outputs

    # Alias
    model.io = SetOf(model.i) 

    # Parameters (define and import the input-output data)
    inputdata = {
                ('OPEX', 'i1') :   681,
                ('OPEX', 'i2') :   559,
                ('OPEX', 'i3') :   836,
                ('OPEX', 'i4') :   7559,
                ('OPEX', 'i5') :   424,
                ('OPEX', 'i6') :   1483,
                ('OPEX', 'i7') :   658,
                ('OPEX', 'i8') :   1433,
                ('OPEX', 'i9') :   850,
                ('OPEX', 'i10') :  1155,
                ('CAPEX', 'i1') :  729,
                ('CAPEX', 'i2') :  673,
                ('CAPEX', 'i3') :  851,
                ('CAPEX', 'i4') :  8384,
                ('CAPEX', 'i5') :  562,
                ('CAPEX', 'i6') :  1587,
                ('CAPEX', 'i7') :  570,
                ('CAPEX', 'i8') :  1311,
                ('CAPEX', 'i9') :  564,
                ('CAPEX', 'i10'):  1108,
                }
    model.x = Param(model.j, model.i, initialize=inputdata)

    outputdata = {
                ('Energy', 'i1'):   75,
                ('Energy', 'i2'):   62,
                ('Energy', 'i3'):   78,
                ('Energy', 'i4'):   683,
                ('Energy', 'i5'):   27,
                ('Energy', 'i6'):   295,
                ('Energy', 'i7'):   44,
                ('Energy', 'i8'):   171,
                ('Energy', 'i9'):   98,
                ('Energy', 'i10'):  203,
                ('Length', 'i1'):   878,
                ('Length', 'i2'):   964,
                ('Length', 'i3'):   676,
                ('Length', 'i4'):   12522,
                ('Length', 'i5'):   697,
                ('Length', 'i6'):   953,
                ('Length', 'i7'):   917,
                ('Length', 'i8'):   1580,
                ('Length', 'i9'):   116,
                ('Length', 'i10'):  740,
                ('Customers', 'i1'):4933,
                ('Customers', 'i2'):6149,
                ('Customers', 'i3'):6098,
                ('Customers', 'i4'):55226,
                ('Customers', 'i5'):1670,
                ('Customers', 'i6'):22949,
                ('Customers', 'i7'):3599,
                ('Customers', 'i8'):11081,
                ('Customers', 'i9'):377,
                ('Customers', 'i10'):10134,
                }
    model.y = Param(model.k, model.i, initialize=outputdata)  

    # Variables
    model.lamda = Var(model.io, model.i, bounds=(0.0, None), doc='envelopment efficiency') 
    model.theta = Var(model.io, doc='intensity variable') 

    # objective
    def objective_rule(model):
        return sum(model.theta[io] for io in model.io)
    model.objective = Objective(rule=objective_rule, sense=minimize, doc='Define objective function')       
    # Constraints
    def output_rule(model, io, k):
        return sum(model.lamda[io, i] * model.y[k, i] for i in model.i) >= model.y[k, io]
    model.output = Constraint(model.io, model.k, rule=output_rule, doc='output constraints')
    def input_rule(model, io, j):
        return (model.theta[io] * model.x[j, io]) >= sum(
             model.lamda[io, i] * model.x[j, i] for i in model.i)
    model.input = Constraint(model.io, model.j, rule=input_rule, doc='input constraints')
    def vrs_rule(model, io):
        return sum(model.lamda[io, i] for i in model.i) == 1
    model.vrs = Constraint(model.io, rule=vrs_rule, doc='VRS constraints')

    # calculate the DEA model 
    from pyomo.opt import SolverFactory
    solver_manager = SolverManagerFactory('neos')
    results = solver_manager.solve(model, opt='cplex')

    # display the estimates
    # efficiency
    model.theta.display()
    # intensity
    model.lamda.display()
    