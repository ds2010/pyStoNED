from . import CNLS

from pyomo.environ import ConcreteModel, Set, Var, Objective, minimize, Constraint
from pyomo.opt import SolverFactory, SolverManagerFactory
from pyomo.core.expr.numvalue import NumericValue
import pandas as pd
import numpy as np


class CNLSDDF(CNLS.CNLS):
    """Convex Nonparametric Least Square with multiple Outputs (DDF formulation)"""

    def __init__(self, y, x, b=None, gy=[1], gx=[1], gb=None, fun='prod'):
        """
        Initialize the CNLSDDF model

        * y: Output variables
        * x: Input variables
        * b: Undesirable output variables
        * gy: Output directional vector
        * gx: Input directional vector
        * gb: Undesirable output directional vector
        * fun = "prod": Production frontier
              = "cost": Cost frontier
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

    def optimize(self, remote=False):
        """Optimize the function by requested method"""
        # TODO(error/warning handling): Check problem status after optimization
        if remote == False:
            print("Estimating the model locally with mosek solver")
            solver = SolverFactory("mosek")
            self.problem_status = solver.solve(self.__model__,
                                               tee=True)
            print(self.problem_status)
            self.optimization_status = 1

        else:
            print("Estimating the model remotely with mosek solver")
            solver = SolverManagerFactory('neos')
            self.problem_status = solver.solve(self.__model__,
                                               tee=True,
                                               opt="mosek")
            print(self.problem_status)
            self.optimization_status = 1

    def __to_1d_list(self, l):
        if type(l) == int or type(l) == float:
            return [l]
        else:
            return l

    def __regression_rule(self):
        """Return the proper regression constraint"""
        if type(self.b) == type(None):
            def regression_rule(model, i):
                return sum(model.gamma[i, k] * self.y[i][k] for k in model.K)\
                    == model.alpha[i]\
                    + sum(model.beta[i, j] * self.x[i][j] for j in model.J)\
                    - model.epsilon[i]

            return regression_rule

        def regression_rule(model, i):
            return sum(model.gamma[i, k] * self.y[i][k] for k in model.K)\
                == model.alpha[i]\
                + sum(model.beta[i, j] * self.x[i][j] for j in model.J)\
                + sum(model.delta[i, l] * self.b[i][l] for l in model.L)\
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
