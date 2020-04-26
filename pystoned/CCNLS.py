"""
@Title   : Corrected Convex Nonparametric Least Squares (C2NLS) 
@Author  : Sheng Dai, Timo Kuosmanen
@Mail    : sheng.dai@aalto.fi (S. Dai); timo.kuosmanen@aalto.fi (T. Kuosmanen)  
@Date    : 2020-04-16
"""

# Import of the pyomo module
from pyomo.environ import *


def ccnls(y, x):

    # number of DMUS
    n = len(y)

    # number of inputs
    if type(x[0]) == int or type(x[0]) == float:
        m = 1
    else:
        m = len(x[0])

    # Creation of a Concrete Model
    model = ConcreteModel()

    # Set
    model.i = Set(initialize=range(n))
    model.j = Set(initialize=range(m))

    # Alias
    model.h = SetOf(model.i)

    # Variables
    model.a = Var(model.i, doc='alpha')
    model.b = Var(model.i, model.j, bounds=(0.0, None), doc='beta')
    model.e = Var(model.i, domain=NegativeReals, doc='residuals')

    # Objective function
    def objective_rule(model):
        return sum(model.e[i] * model.e[i] for i in model.i)

    model.objective = Objective(rule=objective_rule, sense=minimize, doc='Define objective function')

    # Constraints
    def reg_rule(model, i):
        arow = x[i]
        return y[i] == model.a[i] + sum(model.b[i, j] * arow[j] for j in model.j) + model.e[i]

    model.reg = Constraint(model.i, rule=reg_rule, doc='regression')

    def concav_rule(model, i, h):
        arow = x[i]
        if i == h:
            return Constraint.Skip
        return model.a[i] + sum(model.b[i, j] * arow[j] for j in model.j) <= model.a[h] + sum(
            model.b[h, j] * arow[j] for j in model.j)

    model.concav = Constraint(model.i, model.h, rule=concav_rule, doc='concavity constraint')

    return model