"""
@Title   : Convex Nonparametric Least Square (CNLS)
@Author  : Sheng Dai, Timo Kuosmanen
@Mail    : sheng.dai@aalto.fi (S. Dai); timo.kuosmanen@aalto.fi (T. Kuosmanen)  
@Date    : 2020-04-16
"""

# Import of the pyomo module
from pyomo.environ import *


def cnls(y, x, cet, fun, rts):
    # cet     = "addi" : Additive composite error term
    #         = "mult" : Multiplicative composite error term
    # fun     = "prod" : production frontier
    #         = "cost" : cost frontier
    # rts     = "vrs"  : variable returns to scale
    #         = "crs"  : constant returns to scale

    # transform data
    x = x.tolist()
    y = y.tolist()

    # number of DMUs
    n = len(y)

    # number of inputs
    if type(x[0]) == int or type(x[0]) == float:
        m = 1
    else:
        m = len(x[0])

    # Creation of a Concrete Model
    model = ConcreteModel()

    if m == 1:

        # Set
        model.i = Set(initialize=range(n))

        # Alias
        model.h = SetOf(model.i)

        # Variables
        model.a = Var(model.i, doc='alpha')
        model.b = Var(model.i, bounds=(0.0, None), doc='beta')
        model.e = Var(model.i, doc='residuals')
        model.f = Var(model.i, bounds=(0.0, None), doc='estimated frontier')

        # Additive composite error term
        if cet == "addi":

            # Objective function
            def objective_rule(model):
                return sum(model.e[i] * model.e[i] for i in model.i)

            model.objective = Objective(rule=objective_rule, sense=minimize, doc='objective function')

            if rts == "vrs":

                # Constraints
                def reg_rule(model, i):
                    return y[i] == model.a[i] + model.b[i] * x[i] + model.e[i]

                model.reg = Constraint(model.i, rule=reg_rule, doc='regression equation')

                # production model
                if fun == "prod":

                    def concav_rule(model, i, h):
                        if i == h:
                            return Constraint.Skip
                        return model.a[i] + model.b[i] * x[i] <= model.a[h] + model.b[h] * x[i]

                    model.concav = Constraint(model.i, model.h, rule=concav_rule, doc='concavity constraint')

                # cost model
                if fun == "cost":

                    def convex_rule(model, i, h):
                        if i == h:
                            return Constraint.Skip
                        return model.a[i] + model.b[i] * x[i] >= model.a[h] + model.b[h] * x[i]

                    model.convex = Constraint(model.i, model.h, rule=convex_rule, doc='convexity constraint')

        # Multiplicative composite error term
        if cet == "mult":

            # Objectivr function
            def objective_rule(model):
                return sum(model.e[i] * model.e[i] for i in model.i)

            model.objective = Objective(rule=objective_rule, sense=minimize, doc='objective function')

            if rts == "vrs":

                # Constraints
                def qreg_rule(model, i):
                    return log(y[i]) == log(model.f[i] + 1) + model.e[i]

                model.qreg = Constraint(model.i, rule=qreg_rule, doc='log-transformed regression equation')

                def qlog_rule(model, i):
                    return model.f[i] == model.a[i] + model.b[i] * x[i] - 1

                model.qlog = Constraint(model.i, rule=qlog_rule, doc='regression function')

                # production model
                if fun == "prod":

                    def qconcav_rule(model, i, h):
                        if i == h:
                            return Constraint.Skip
                        return model.a[i] + model.b[i] * x[i] <= model.a[h] + model.b[h] * x[i]

                    model.qconcav = Constraint(model.i, model.h, rule=qconcav_rule, doc='concavity constraint')

                # cost model
                if fun == "cost":

                    def qconvex_rule(model, i, h):
                        if i == h:
                            return Constraint.Skip
                        return model.a[i] + model.b[i] * x[i] >= model.a[h] + model.b[h] * x[i]

                    model.qconvex = Constraint(model.i, model.h, rule=qconvex_rule, doc='convexity constraint')

            if rts == "crs":

                # Constraints
                def qreg_rule(model, i):
                    return log(y[i]) == log(model.f[i] + 1) + model.e[i]

                model.qreg = Constraint(model.i, rule=qreg_rule, doc='log-transformed regression equation')

                def qlog_rule(model, i):
                    return model.f[i] == model.b[i] * x[i] - 1

                model.qlog = Constraint(model.i, rule=qlog_rule, doc='regression function')

                # production model
                if fun == "prod":

                    def qconcav_rule(model, i, h):
                        if i == h:
                            return Constraint.Skip
                        return model.b[i] * x[i] <= model.b[h] * x[i]

                    model.qconcav = Constraint(model.i, model.h, rule=qconcav_rule, doc='concavity constraint')

                # cost model
                if fun == "cost":

                    def qconvex_rule(model, i, h):
                        if i == h:
                            return Constraint.Skip
                        return model.b[i] * x[i] >= model.b[h] * x[i]

                    model.qconvex = Constraint(model.i, model.h, rule=qconvex_rule, doc='convexity constraint')

    if m > 1:

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
        if cet == "addi":

            # Objective function
            def objective_rule(model):
                return sum(model.e[i] * model.e[i] for i in model.i)

            model.objective = Objective(rule=objective_rule, sense=minimize, doc='objective function')

            if rts == "vrs":

                # Constraints
                def reg_rule(model, i):
                    arow = x[i]
                    return y[i] == model.a[i] + sum(model.b[i, j] * arow[j] for j in model.j) + model.e[i]

                model.reg = Constraint(model.i, rule=reg_rule, doc='regression equation')

                # production model
                if fun == "prod":

                    def concav_rule(model, i, h):
                        arow = x[i]
                        if i == h:
                            return Constraint.Skip
                        return model.a[i] + sum(model.b[i, j] * arow[j] for j in model.j) <= model.a[h] + \
                               sum(model.b[h, j] * arow[j] for j in model.j)

                    model.concav = Constraint(model.i, model.h, rule=concav_rule, doc='concavity constraint')

                # cost model
                if fun == "cost":

                    def convex_rule(model, i, h):
                        arow = x[i]
                        if i == h:
                            return Constraint.Skip
                        return model.a[i] + sum(model.b[i, j] * arow[j] for j in model.j) >= model.a[h] + \
                               sum(model.b[h, j] * arow[j] for j in model.j)

                    model.convex = Constraint(model.i, model.h, rule=convex_rule, doc='convexity constraint')

        # Multiplicative composite error term
        if cet == "mult":

            # Objectivr function
            def objective_rule(model):
                return sum(model.e[i] * model.e[i] for i in model.i)

            model.objective = Objective(rule=objective_rule, sense=minimize, doc='objective function')

            if rts == "vrs":

                # Constraints
                def qreg_rule(model, i):
                    return log(y[i]) == log(model.f[i] + 1) + model.e[i]

                model.qreg = Constraint(model.i, rule=qreg_rule, doc='log-transformed regression equation')

                def qlog_rule(model, i):
                    arow = x[i]
                    return model.f[i] == model.a[i] + sum(model.b[i, j] * arow[j] for j in model.j) - 1

                model.qlog = Constraint(model.i, rule=qlog_rule, doc='regression function')

                # production model
                if fun == "prod":

                    def qconcav_rule(model, i, h):
                        arow = x[i]
                        if i == h:
                            return Constraint.Skip
                        return model.a[i] + sum(model.b[i, j] * arow[j] for j in model.j) <= model.a[h] + \
                               sum(model.b[h, j] * arow[j] for j in model.j)

                    model.qconcav = Constraint(model.i, model.h, rule=qconcav_rule, doc='concavity constraint')

                # cost model
                if fun == "cost":

                    def qconvex_rule(model, i, h):
                        arow = x[i]
                        if i == h:
                            return Constraint.Skip
                        return model.a[i] + sum(model.b[i, j] * arow[j] for j in model.j) >= model.a[h] + \
                               sum(model.b[h, j] * arow[j] for j in model.j)

                    model.qconvex = Constraint(model.i, model.h, rule=qconvex_rule, doc='convexity constraint')

            if rts == "crs":

                # Constraints
                def qreg_rule(model, i):
                    return log(y[i]) == log(model.f[i] + 1) + model.e[i]

                model.qreg = Constraint(model.i, rule=qreg_rule, doc='log-transformed regression equation')

                def qlog_rule(model, i):
                    arow = x[i]
                    return model.f[i] == sum(model.b[i, j] * arow[j] for j in model.j) - 1

                model.qlog = Constraint(model.i, rule=qlog_rule, doc='regression function')

                # production model
                if fun == "prod":

                    def qconcav_rule(model, i, h):
                        arow = x[i]
                        if i == h:
                            return Constraint.Skip
                        return sum(model.b[i, j] * arow[j] for j in model.j) <= \
                               sum(model.b[h, j] * arow[j] for j in model.j)

                    model.qconcav = Constraint(model.i, model.h, rule=qconcav_rule, doc='concavity constraint')

                # cost model
                if fun == "cost":

                    def qconvex_rule(model, i, h):
                        arow = x[i]
                        if i == h:
                            return Constraint.Skip
                        return sum(model.b[i, j] * arow[j] for j in model.j) >= \
                               sum(model.b[h, j] * arow[j] for j in model.j)

                    model.qconvex = Constraint(model.i, model.h, rule=qconvex_rule, doc='convexity constraint')

    return model