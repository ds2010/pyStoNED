"""
@Title   : Convex Nonparametric Least Square (CNLS)
@Author  : Sheng Dai, Timo Kuosmanen
@Mail    : sheng.dai@aalto.fi (S. Dai); timo.kuosmanen@aalto.fi (T. Kuosmanen)  
@Date    : 2020-04-16
"""

# Import of the pyomo module
from pyomo.environ import *


def cnls(y, x, crt, func, pps):
    # crt     = "addi" : Additive composite error term
    #         = "mult" : Multiplicative composite error term
    # func    = "prod" : production frontier
    #         = "cost" : cost frontier
    # pps     = "vrs"  : variable returns to scale production possibility sets (pps)
    #         = "crs"  : constant returns to scale pps

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
    model.e = Var(model.i, doc='residuals')
    model.f = Var(model.i, bounds=(0.0, None), doc='estimated frontier')

    # Additive composite error term
    if crt == "addi":

        # Objective function
        def objective_rule(model):
            return sum(model.e[i] * model.e[i] for i in model.i)

        model.objective = Objective(rule=objective_rule, sense=minimize, doc='Define objective function')

        if pps == "vrs":

            # Constraints
            def reg_rule(model, i):
                arow = x[i]
                return y[i] == model.a[i] + sum(model.b[i, j] * arow[j] for j in model.j) + model.e[i]

            model.reg = Constraint(model.i, rule=reg_rule, doc='regression')

            # production model
            if func == "prod":

                def concav_rule(model, i, h):
                    arow = x[i]
                    if i == h:
                        return Constraint.Skip
                    return model.a[i] + sum(model.b[i, j] * arow[j] for j in model.j) <= model.a[h] + sum(
                        model.b[h, j] * arow[j] for j in model.j)

                model.concav = Constraint(model.i, model.h, rule=concav_rule, doc='concavity constraint')

            # cost model
            if func == "cost":

                def concav_rule(model, i, h):
                    arow = x[i]
                    if i == h:
                        return Constraint.Skip
                    return model.a[i] + sum(model.b[i, j] * arow[j] for j in model.j) >= model.a[h] + sum(
                        model.b[h, j] * arow[j] for j in model.j)

                model.concav = Constraint(model.i, model.h, rule=concav_rule, doc='concavity constraint')

    # Multiplicative composite error term
    if crt == "mult":

        # Objectivr function
        def objective_rule(model):
            return sum(model.e[i] * model.e[i] for i in model.i)

        model.objective = Objective(rule=objective_rule, sense=minimize, doc='Define objective function')

        if pps == "vrs":

            # Constraints
            def qreg_rule(model, i):
                return log(y[i]) == log(model.f[i] + 1) + model.e[i]

            model.qreg = Constraint(model.i, rule=qreg_rule, doc='log-transformed regression')

            def qlog_rule(model, i):
                arow = x[i]
                return model.f[i] == model.a[i] + sum(model.b[i, j] * arow[j] for j in model.j) - 1

            model.qlog = Constraint(model.i, rule=qlog_rule, doc='cost function')

            # production model
            if func == "prod":
                def qconcav_rule(model, i, h):
                    arow = x[i]
                    if i == h:
                        return Constraint.Skip
                    return model.a[i] + sum(model.b[i, j] * arow[j] for j in model.j) <= model.a[h] + sum(
                        model.b[h, j] * arow[j] for j in model.j)

                model.qconcav = Constraint(model.i, model.h, rule=qconcav_rule, doc='concavity constraint')

            # cost model
            if func == "cost":

                def qconcav_rule(model, i, h):
                    arow = x[i]
                    if i == h:
                        return Constraint.Skip
                    return model.a[i] + sum(model.b[i, j] * arow[j] for j in model.j) >= model.a[h] + sum(
                        model.b[h, j] * arow[j] for j in model.j)

                model.qconcav = Constraint(model.i, model.h, rule=qconcav_rule, doc='concavity constraint')

        if pps == "crs":

            # Constraints
            def qreg_rule(model, i):
                return log(y[i]) == log(model.f[i] + 1) + model.e[i]

            model.qreg = Constraint(model.i, rule=qreg_rule, doc='log-transformed regression')

            def qlog_rule(model, i):
                arow = x[i]
                return model.f[i] == sum(model.b[i, j] * arow[j] for j in model.j) - 1

            model.qlog = Constraint(model.i, rule=qlog_rule, doc='cost function')

            # production model
            if func == "prod":
                def qconcav_rule(model, i, h):
                    arow = x[i]
                    if i == h:
                        return Constraint.Skip
                    return sum(model.b[i, j] * arow[j] for j in model.j) <= sum(
                        model.b[h, j] * arow[j] for j in model.j)

                model.qconcav = Constraint(model.i, model.h, rule=qconcav_rule, doc='concavity constraint')

            # cost model
            if func == "cost":

                def qconcav_rule(model, i, h):
                    arow = x[i]
                    if i == h:
                        return Constraint.Skip
                    return sum(model.b[i, j] * arow[j] for j in model.j) >= sum(
                        model.b[h, j] * arow[j] for j in model.j)

                model.qconcav = Constraint(model.i, model.h, rule=qconcav_rule, doc='concavity constraint')

    return model
