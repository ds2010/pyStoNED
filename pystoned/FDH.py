"""
@Title   : Free Disposal Hull (FDH)
@Author  : Sheng Dai, Timo Kuosmanen
@Mail    : sheng.dai@aalto.fi (S. Dai); timo.kuosmanen@aalto.fi (T. Kuosmanen)
@Date    : 2020-06-18
"""

# Import of the pyomo module
from pyomo.environ import *
import numpy as np


def fdh(y, x, orient):
    # orient  = "io" : input orientation
    #         = "oo" : output orientation

    # transform data
    x = x.tolist()
    y = y.tolist()

    # number of DMUs
    n = len(y)

    # number of inputs
    if type(x[0]) == int or type(x[0]) == float:
        m = 1
        x = np.asmatrix(np.array(x)).tolist()
    else:
        m = len(x[0])
        x = np.array(x).T.tolist()

        # number of outputs
    if type(y[0]) == int or type(y[0]) == float:
        p = 1
        y = np.asmatrix(np.array(y)).tolist()
    else:
        p = len(y[0])
        y = np.array(y).T.tolist()

        # Creation of a Concrete Model
    model = ConcreteModel()

    # Set
    model.i = Set(initialize=range(n))
    model.j = Set(initialize=range(m))
    model.k = Set(initialize=range(p))

    # Alias
    model.io = SetOf(model.i)

    # Variables
    model.theta = Var(model.io, doc='efficiency')
    model.lamda = Var(model.io, model.i, within=Binary, doc='intensity variables')

    if orient == "io":
        
        # objective
        def objective_rule(model):
            return sum(model.theta[io] for io in model.io)

        model.objective = Objective(rule=objective_rule, sense=minimize, doc='objective function')

        # Constraints
        def input_rule(model, io, j):
            arow = x[j]
            return (model.theta[io] * arow[io]) >= sum(model.lamda[io, i] * arow[i] for i in model.i)

        model.input = Constraint(model.io, model.j, rule=input_rule, doc='input constraints')

        def output_rule(model, io, k):
            brow = y[k]
            return sum(model.lamda[io, i] * brow[i] for i in model.i) >= brow[io]

        model.output = Constraint(model.io, model.k, rule=output_rule, doc='output constraints')

        def vrs_rule(model, io):
            return sum(model.lamda[io, i] for i in model.i) == 1

        model.vrs = Constraint(model.io, rule=vrs_rule, doc='VRS constraints')
           

    if orient == "oo":

       # objective
       def objective_rule(model):
           return sum(model.theta[io] for io in model.io)

       model.objective = Objective(rule=objective_rule, sense=maximize, doc='objective function')

       # Constraints
       def input_rule(model, io, j):
           arow = x[j]
           return sum(model.lamda[io, i] * arow[i] for i in model.i) <= arow[io]

       model.input = Constraint(model.io, model.j, rule=input_rule, doc='input constraints')

       def output_rule(model, io, k):
           brow = y[k]
           return model.theta[io] * brow[io] <= sum(model.lamda[io, i] * brow[i] for i in model.i)

       model.output = Constraint(model.io, model.k, rule=output_rule, doc='output constraints')

       def vrs_rule(model, io):
           return sum(model.lamda[io, i] for i in model.i) == 1

       model.vrs = Constraint(model.io, rule=vrs_rule, doc='VRS constraints')


    return model
