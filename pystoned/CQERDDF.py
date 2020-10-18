from . import CNLSDDF, CQER
from pyomo.environ import ConcreteModel, Set, Var, Objective, minimize, Constraint
from pyomo.core.expr.numvalue import NumericValue


class CQRDDF(CNLSDDF.CNLSDDF, CQER.CQR):
    """Convex quantile regression with multiple Outputs (DDF formulation)"""

    def __init__(self, y, x, b=None, gy=[1], gx=[1], gb=None, fun='prod', tau=0.9):
        """
            y : Output variables
            x : Input variables
            b : Undesirable output variables
            gy : Output directional vector
            gx : Input directional vector
            gb : Undesirable output directional vector
            fun  = "prod" : Production frontier
                 = "cost" : Cost frontier
        """

        # TODO(error/warning handling): Check the configuration of the model exist
        self.x = x.tolist()
        self.y = y.tolist()
        self.b = b
        self.tau = tau
        self.fun = fun

        self.gy = self._CNLSDDF__to_1d_list(gy)
        self.gx = self._CNLSDDF__to_1d_list(gx)
        self.gb = self._CNLSDDF__to_1d_list(gb)

        if type(self.x[0]) != list:
            self.x = []
            for x_value in x.tolist():
                self.x.append([x_value])

        if type(self.y[0]) != list:
            self.y = []
            for y_value in y.tolist():
                self.y.append([y_value])

        self.__model__ = ConcreteModel()

        # Initialize the sets
        self.__model__.I = Set(initialize=range(len(self.y)))
        self.__model__.J = Set(initialize=range(len(self.x[0])))
        self.__model__.K = Set(initialize=range(len(self.y[0])))

        # Initialize the variables
        self.__model__.alpha = Var(self.__model__.I, doc='alpha')
        self.__model__.beta = Var(
            self.__model__.I, self.__model__.J, bounds=(0.0, None), doc='beta')
        self.__model__.gamma = Var(
            self.__model__.I, self.__model__.K, bounds=(0.0, None), doc='gamma')

        self.__model__.epsilon = Var(self.__model__.I, doc='residuals')
        self.__model__.epsilon_plus = Var(
            self.__model__.I, bounds=(0.0, None), doc='positive error term')
        self.__model__.epsilon_minus = Var(
            self.__model__.I, bounds=(0.0, None), doc='negative error term')

        if type(self.b) != type(None):
            self.b = b.tolist()
            self.gb = self._CNLSDDF__to_1d_list(gb)

            if type(self.b[0]) != list:
                self.b = []
                for b_value in b.tolist():
                    self.b.append([b_value])

            self.__model__.L = Set(initialize=range(len(self.b[0])))
            self.__model__.delta = Var(
                self.__model__.I, self.__model__.L, bounds=(0.0, None), doc='delta')

        self.__model__.objective = Objective(rule=self._CQR__objective_rule(),
                                             sense=minimize,
                                             doc='objective function')

        self.__model__.error_decomposition = Constraint(self.__model__.I,
                                                        rule=self._CQR__error_decomposition(),
                                                        doc='decompose error term')

        self.__model__.regression_rule = Constraint(self.__model__.I,
                                                    rule=self.__regression_rule(),
                                                    doc='regression equation')

        self.__model__.translation_rule = Constraint(self.__model__.I,
                                                     rule=self._CNLSDDF__translation_property(),
                                                     doc='translation property')

        self.__model__.afriat_rule = Constraint(self.__model__.I,
                                                self.__model__.I,
                                                rule=self.__afriat_rule(),
                                                doc='afriat inequality')

        # Optimize model
        self.optimization_status = 0
        self.problem_status = 0

    def __regression_rule(self):
        """Return the proper regression constraint"""
        if type(self.b) == type(None):
            def regression_rule(model, i):
                return sum(model.gamma[i, k] * self.y[i][k] for k in model.K)\
                    == model.alpha[i]\
                    + sum(model.beta[i, j] * self.x[i][j] for j in model.J)\
                    + model.epsilon[i]

            return regression_rule

        def regression_rule(model, i):
            return sum(model.gamma[i, k] * self.y[i][k] for k in model.K)\
                == model.alpha[i]\
                + sum(model.beta[i, j] * self.x[i][j] for j in model.J)\
                + sum(model.delta[i, l] * self.b[i][l] for l in model.L)\
                + model.epsilon[i]

        return regression_rule

    def __afriat_rule(self):
        """Return the proper afriat inequality constraint"""
        if self.fun == "prod":
            __operator = NumericValue.__le__
        elif self.fun == "cost":
            __operator = NumericValue.__ge__

        if type(self.b) == type(None):
            def afriat_rule(model, i, h):
                if i == h:
                    return Constraint.Skip
                return __operator(model.alpha[i]
                                  + sum(model.beta[i, j] * self.x[i][j]
                                        for j in model.J)
                                  - sum(model.gamma[i, k] * self.y[i][k]
                                        for k in model.K),
                                  model.alpha[h]
                                  + sum(model.beta[h, j] * self.x[i][j]
                                        for j in model.J)
                                  - sum(model.gamma[h, k] * self.y[i][k] for k in model.K))

            return afriat_rule

        def afriat_rule(model, i, h):
            if i == h:
                return Constraint.Skip
            return __operator(model.alpha[i]
                              + sum(model.beta[i, j] * self.x[i][j]
                                    for j in model.J)
                              + sum(model.delta[i, l] * self.b[i][l]
                                    for l in model.L)
                              - sum(model.gamma[i, k] * self.y[i][k]
                                    for k in model.K),
                              model.alpha[h]
                              + sum(model.beta[h, j] * self.x[i][j]
                                    for j in model.J)
                              + sum(model.delta[h, l] * self.b[i][l]
                                    for l in model.L)
                              - sum(model.gamma[h, k] * self.y[i][k] for k in model.K))

        return afriat_rule


class CERDDF(CQRDDF):
    """Convex expectile regression with multiple Outputs (DDF formulation)"""

    def __init__(self, y, x,  b=None, gy=[1], gx=[1], gb=None, fun='prod', tau=0.9):
        """
            y : Output variables
            x : Input variables
            b : Undesirable output variables
            gy : Output directional vector
            gx : Input directional vector
            gb : Undesirable output directional vector
            fun  = "prod" : Production frontier
                 = "cost" : Cost frontier
        """
        super().__init__(y, x,  b, gy, gx, gb, fun, tau)
        self.__model__.objective.deactivate()
        self.__model__.squared_objective = Objective(
            rule=self.__squared_objective_rule(), sense=minimize, doc='squared objective rule')

    def __squared_objective_rule(self):
        def squared_objective_rule(model):
            return self.tau * sum(model.epsilon_plus[i] ** 2 for i in model.I)\
                + (1 - self.tau) * \
                sum(model.epsilon_minus[i] ** 2 for i in model.I)
        return squared_objective_rule
