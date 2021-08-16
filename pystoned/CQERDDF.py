# import dependencies
from pyomo.environ import ConcreteModel, Set, Var, Objective, minimize, Constraint
from pyomo.core.expr.numvalue import NumericValue
from .constant import FUN_PROD, FUN_COST, RTS_VRS
from . import CNLSDDF, CQER
from .utils import tools


class CQRDDF(CNLSDDF.CNLSDDF, CQER.CQR):
    """Convex quantile regression with DDF formulation
    """

    def __init__(self, y, x, b=None, gy=[1], gx=[1], gb=None, fun=FUN_PROD, tau=0.5):
        """CQR DDF 

        Args:
            y (float): output variable.
            x (float): input variables.
            b (float), optional): undesirable output variables. Defaults to None.
            gy (list, optional): output directional vector. Defaults to [1].
            gx (list, optional): input directional vector. Defaults to [1].
            gb (list, optional): undesirable output directional vector. Defaults to None.
            fun (String, optional): FUN_PROD (production frontier) or FUN_COST (cost frontier). Defaults to FUN_PROD.
            tau (float, optional): quantile. Defaults to 0.5.
        """
        # TODO(error/warning handling): Check the configuration of the model exist
        self.y, self.x, self.b, self.gy, self.gx, self.gb = tools.assert_valid_direciontal_data(y,x,b,gy,gx,gb)
        self.tau = tau
        self.fun = fun
        self.rts = RTS_VRS
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
                return sum(model.gamma[i, k] * self.y[i][k] for k in model.K) \
                    == model.alpha[i] \
                    + sum(model.beta[i, j] * self.x[i][j] for j in model.J) \
                    + model.epsilon[i]

            return regression_rule

        def regression_rule(model, i):
            return sum(model.gamma[i, k] * self.y[i][k] for k in model.K) \
                == model.alpha[i] \
                + sum(model.beta[i, j] * self.x[i][j] for j in model.J) \
                + sum(model.delta[i, l] * self.b[i][l] for l in model.L) \
                + model.epsilon[i]

        return regression_rule

    def __afriat_rule(self):
        """Return the proper afriat inequality constraint"""
        if self.fun == FUN_PROD:
            __operator = NumericValue.__le__
        elif self.fun == FUN_COST:
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
    """Convex expectile regression with DDF formulation
    """

    def __init__(self, y, x, b=None, gy=[1], gx=[1], gb=None, fun=FUN_PROD, tau=0.5):
        """CER DDF 

        Args:
            y (float): output variable.
            x (float): input variables.
            b (float), optional): undesirable output variables. Defaults to None.
            gy (list, optional): output directional vector. Defaults to [1].
            gx (list, optional): input directional vector. Defaults to [1].
            gb (list, optional): undesirable output directional vector. Defaults to None.
            fun (String, optional): FUN_PROD (production frontier) or FUN_COST (cost frontier). Defaults to FUN_PROD.
            tau (float, optional): expectile. Defaults to 0.5.
        """
        super().__init__(y, x, b, gy, gx, gb, fun, tau)
        self.__model__.objective.deactivate()
        self.__model__.squared_objective = Objective(
            rule=self.__squared_objective_rule(), sense=minimize, doc='squared objective rule')

    def __squared_objective_rule(self):
        def squared_objective_rule(model):
            return self.tau * sum(model.epsilon_plus[i] ** 2 for i in model.I) \
                + (1 - self.tau) * \
                sum(model.epsilon_minus[i] ** 2 for i in model.I)

        return squared_objective_rule
