"""
@Title   : Convex Nonparametric Least Square with multiple Outputs (without undesirable outputs and DDF formulation)
@Author  : Sheng Dai, Timo Kuosmanen
@Mail    : sheng.dai@aalto.fi (S. Dai); timo.kuosmanen@aalto.fi (T. Kuosmanen)
@Date    : 2020-04-25
"""

# Import of the pyomo module
from pyomo.environ import *
from . import directV
import numpy as np


def cnlsddf(y, x, func, gx, gy):
    # func    = "prod" : production frontier
    #         = "cost" : cost frontier

    # number of DMUS
    n = len(y)

    # number of inputs
    if type(x[0]) == int or type(x[0]) == float:
        m = 1
    else:
        m = len(x[0])

    # number of outputs
    if type(y[0]) == int or type(y[0]) == float:
        p = 1
    else:
        p = len(y[0])

    # identity matrix
    id = np.repeat(1, n)
    id = id.tolist()

    # directional vectors
    gx = directV.dv(gx, gy, n, m, p)[0]
    gy = directV.dv(gx, gy, n, m, p)[2]

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
    model.e = Var(model.i, doc='residuals')
    model.f = Var(model.i, bounds=(0.0, None), doc='estimated frontier')

    if p == 1:

        # Variables
        model.g = Var(model.i, bounds=(0.0, None), doc='gamma')

        # Objective function
        def objective_rule(model):
            return sum(model.e[i] * model.e[i] for i in model.i)

        model.objective = Objective(rule=objective_rule, sense=minimize, doc='Define objective function')

        # Constraints
        def reg_rule(model, i):
            arow = x[i]
            return model.g[i] * y[i] == model.a[i] + sum(model.b[i, j] * arow[j] for j in model.j) - model.e[i]

        model.reg = Constraint(model.i, rule=reg_rule, doc='regression')

        def trans_rule(model, i):
            erow = gx[i]
            return sum(model.b[i, j] * erow[j] for j in model.j) + model.g[i] * gy[i] == id[i]

        model.trans = Constraint(model.i, rule=trans_rule, doc='translation property')

        # production model
        if func == "prod":

            def concav_rule(model, i, h):
                arow = x[i]
                if i == h:
                    return Constraint.Skip
                return model.a[i] + sum(model.b[i, j] * arow[j] for j in model.j) - model.g[i] * y[
                       i] <= model.a[h] + sum(model.b[h, j] * arow[j] for j in model.j) - model.g[h] * y[i]

            model.concav = Constraint(model.i, model.h, rule=concav_rule, doc='concavity constraint')

        # cost model
        if func == "cost":

            def concav_rule(model, i, h):
                arow = x[i]
                if i == h:
                    return Constraint.Skip
                return model.a[i] + sum(model.b[i, j] * arow[j] for j in model.j) - model.g[i] * y[
                       i] >= model.a[h] + sum(model.b[h, j] * arow[j] for j in model.j) - model.g[h] * y[i]

            model.concav = Constraint(model.i, model.h, rule=concav_rule, doc='concavity constraint')

    if p > 1:

        # Set
        model.k = Set(initialize=range(p))

        # Variables
        model.g = Var(model.i, model.k, bounds=(0.0, None), doc='gamma')

        # Objective function
        def objective_rule(model):
            return sum(model.e[i] * model.e[i] for i in model.i)

        model.objective = Objective(rule=objective_rule, sense=minimize, doc='Define objective function')

        # Constraints
        def reg_rule(model, i):
            arow = x[i]
            brow = y[i]
            return sum(model.g[i, k] * brow[k] for k in model.k) == model.a[i] + sum(
                   model.b[i, j] * arow[j] for j in model.j) - model.e[i]

        model.reg = Constraint(model.i, rule=reg_rule, doc='regression')

        def trans_rule(model, i):
            erow = gx[i]
            frow = gy[i]
            return sum(model.b[i, j] * erow[j] for j in model.j) + sum(model.g[i, k] * frow[k] for k in model.k) == id[i]

        model.trans = Constraint(model.i, rule=trans_rule, doc='translation property')

        # production model
        if func == "cost":

            def concav_rule(model, i, h):
                arow = x[i]
                brow = y[i]
                if i == h:
                    return Constraint.Skip
                return model.a[i] + sum(model.b[i, j] * arow[j] for j in model.j) - \
                       sum(model.g[i, k] * brow[k] for k in model.k) <= model.a[h] + sum(
                       model.b[i, j] * arow[j] for j in model.j) - sum(model.g[h, k] * brow[k] for k in model.k)

            model.concav = Constraint(model.i, model.h, rule=concav_rule, doc='concavity constraint')

        # cost model
        if func == "cost":

            def concav_rule(model, i, h):
                arow = x[i]
                brow = y[i]
                if i == h:
                    return Constraint.Skip
                return model.a[i] + sum(model.b[i, j] * arow[j] for j in model.j)  - \
                       sum(model.g[i, k] * brow[k] for k in model.k) >= model.a[h] + sum(
                       model.b[i, j] * arow[j] for j in model.j) - sum(model.g[h, k] * brow[k] for k in model.k)

            model.concav = Constraint(model.i, model.h, rule=concav_rule, doc='concavity constraint')

    return model