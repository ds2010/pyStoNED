# import dependencies
from pyomo.environ import ConcreteModel, Set, Var, Objective, minimize, Constraint
from pyomo.opt import SolverFactory, SolverManagerFactory
from pyomo.core.expr.numvalue import NumericValue
import pandas as pd
import numpy as np

from . import CNLS
from .constant import CET_ADDI, FUN_COST, FUN_PROD, OPT_DEFAULT, OPT_LOCAL
from .utils import tools


class CNLSDDF(CNLS.CNLS):
    """Convex Nonparametric Least Square with DDF formulation
    """

    def __init__(self, y, x, b=None, gy=[1], gx=[1], gb=None, fun=FUN_PROD):
        """CNLS DDF model

        Args:
            y (float): output variable.
            x (float): input variables.
            b (float), optional): undesirable output variables. Defaults to None.
            gy (list, optional): output directional vector. Defaults to [1].
            gx (list, optional): input directional vector. Defaults to [1].
            gb (list, optional): undesirable output directional vector. Defaults to None.
            fun (String, optional): FUN_PROD (production frontier) or FUN_COST (cost frontier). Defaults to FUN_PROD.
        """
        # TODO(error/warning handling): Check the configuration of the model exist
        self.y = y.tolist()
        self.x = x.tolist()
        self.b = b
        self.gy = self.__to_1d_list(gy)
        self.gx = self.__to_1d_list(gx)
        self.gb = self.__to_1d_list(gb)

        self.fun = fun

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
        self.__model__.epsilon = Var(self.__model__.I, doc='residuals')
        self.__model__.gamma = Var(
            self.__model__.I, self.__model__.K, bounds=(0.0, None), doc='gamma')

        if type(self.b) != type(None):
            self.b = b.tolist()
            self.gb = self.__to_1d_list(gb)

            if type(self.b[0]) != list:
                self.b = []
                for b_value in b.tolist():
                    self.b.append([b_value])

            self.__model__.L = Set(initialize=range(len(self.b[0])))
            self.__model__.delta = Var(
                self.__model__.I, self.__model__.L, bounds=(0.0, None), doc='delta')

        # Setup the objective function and constraints
        self.__model__.objective = Objective(rule=self._CNLS__objective_rule(),
                                             sense=minimize,
                                             doc='objective function')
        self.__model__.regression_rule = Constraint(self.__model__.I,
                                                    rule=self.__regression_rule(),
                                                    doc='regression equation')
        self.__model__.translation_rule = Constraint(self.__model__.I,
                                                     rule=self.__translation_property(),
                                                     doc='translation property')
        self.__model__.afriat_rule = Constraint(self.__model__.I,
                                                self.__model__.I,
                                                rule=self.__afriat_rule(),
                                                doc='afriat inequality')

        # Optimize model
        self.optimization_status = 0
        self.problem_status = 0

    def optimize(self, email=OPT_LOCAL, solver=OPT_DEFAULT):
        """Optimize the function by requested method

        Args:
            email (string): The email address for remote optimization. It will optimize locally if OPT_LOCAL is given.
            solver (string): The solver chosen for optimization. It will optimize with default solver if OPT_DEFAULT is given.
        """
        # TODO(error/warning handling): Check problem status after optimization
        self.problem_status, self.optimization_status = tools.optimize_model(
            self.__model__, email, CET_ADDI, solver)

    def __to_1d_list(self, l):
        if type(l) == int or type(l) == float:
            return [l]
        else:
            return l

    def __regression_rule(self):
        """Return the proper regression constraint"""
        if type(self.b) == type(None):
            def regression_rule(model, i):
                return sum(model.gamma[i, k] * self.y[i][k] for k in model.K) \
                    == model.alpha[i] \
                    + sum(model.beta[i, j] * self.x[i][j] for j in model.J) \
                    - model.epsilon[i]

            return regression_rule

        def regression_rule(model, i):
            return sum(model.gamma[i, k] * self.y[i][k] for k in model.K) \
                == model.alpha[i] \
                + sum(model.beta[i, j] * self.x[i][j] for j in model.J) \
                + sum(model.delta[i, l] * self.b[i][l] for l in model.L) \
                - model.epsilon[i]

        return regression_rule

    def __translation_property(self):
        """Return the proper translation property"""
        if type(self.b) == type(None):
            def translation_rule(model, i):
                return sum(model.beta[i, j] * self.gx[j] for j in model.J) \
                    + sum(model.gamma[i, k] * self.gy[k] for k in model.K) == 1

            return translation_rule

        def translation_rule(model, i):
            return sum(model.beta[i, j] * self.gx[j] for j in model.J) \
                + sum(model.gamma[i, k] * self.gy[k] for k in model.K) \
                + sum(model.delta[i, l] * self.gb[l] for l in model.L) == 1

        return translation_rule

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
            return __operator(model.epsilon[i],
                              model.alpha[h]
                              + sum(model.beta[h, j] * self.x[i][j]
                                    for j in model.J)
                              + sum(model.delta[h, l] * self.b[i][l]
                                    for l in model.L)
                              - sum(model.gamma[h, k] * self.y[i][k] for k in model.K))

        return afriat_rule

    def display_gamma(self):
        """Display gamma value"""
        if self.optimization_status == 0:
            print("Model isn't optimized. Use optimize() method to estimate the model.")
            return False
        self.__model__.gamma.display()

    def display_delta(self):
        """Display delta value"""
        if self.optimization_status == 0:
            print("Model isn't optimized. Use optimize() method to estimate the model.")
            return False
        if type(self.b) == type(None):
            # TODO(warning handling): replace print to warning
            print("No undesirable output given.")
            return False

        self.__model__.delta.display()

    def get_gamma(self):
        """Return gamma value by array"""
        if self.optimization_status == 0:
            print("Model isn't optimized. Use optimize() method to estimate the model.")
            return False
        gamma = np.asarray([i + tuple([j]) for i, j in zip(list(self.__model__.gamma),
                                                           list(self.__model__.gamma[:, :].value))])
        gamma = pd.DataFrame(gamma, columns=['Name', 'Key', 'Value'])
        gamma = gamma.pivot(index='Name', columns='Key', values='Value')
        return gamma.to_numpy()

    def get_delta(self):
        """Return delta value by array"""
        if self.optimization_status == 0:
            print("Model isn't optimized. Use optimize() method to estimate the model.")
            return False
        if type(self.b) == type(None):
            # TODO(warning handling): replace print to warning
            print("No undesirable output given.")
            return False

        delta = np.asarray([i + tuple([j]) for i, j in zip(list(self.__model__.delta),
                                                           list(self.__model__.delta[:, :].value))])
        delta = pd.DataFrame(delta, columns=['Name', 'Key', 'Value'])
        delta = delta.pivot(index='Name', columns='Key', values='Value')
        return delta.to_numpy()
