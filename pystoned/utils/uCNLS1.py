# import dependencies
from pyomo.environ import ConcreteModel, Set, Var, Objective, minimize, Constraint
import numpy as np

from ..constant import CET_ADDI, OPT_DEFAULT, OPT_LOCAL
from .tools import optimize_model


class uCNLS1:
    """initial Group-VC-added CNLS (CNLS+G) model
    """

    def __init__(self, y, x):
        """uCNLS model 1

        Args:
            y (float): output variable.
            x (float): input variable.
        """
        # TODO(error/warning handling): Check the configuration of the model exist
        self.x = x
        self.y = y
        self.cet = CET_ADDI

        # Initialize the CNLS model
        self.__model__ = ConcreteModel()

        # Initialize the sets
        self.__model__.I = Set(initialize=range(len(self.y)))

        # Initialize the variables
        self.__model__.alpha = Var(self.__model__.I, doc='alpha')
        self.__model__.beta = Var(self.__model__.I,
                                  bounds=(0.0, None),
                                  doc='beta')
        self.__model__.epsilon = Var(self.__model__.I, doc='residual')
        self.__model__.frontier = Var(self.__model__.I,
                                      bounds=(0.0, None),
                                      doc='estimated frontier')

        # Setup the objective function and constraints
        self.__model__.objective = Objective(rule=self.__objective_rule(),
                                             sense=minimize,
                                             doc='objective function')
        self.__model__.regression_rule = Constraint(self.__model__.I,
                                                    rule=self.__regression_rule(),
                                                    doc='regression equation')
        self.__model__.afriat_rule1 = Constraint(self.__model__.I,
                                                 rule=self.__afriat_rule1(),
                                                 doc='afriat inequality: beta')
        self.__model__.afriat_rule2 = Constraint(self.__model__.I,
                                                 rule=self.__afriat_rule2(),
                                                 doc='afriat inequality: alpha')

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
        self.problem_status, self.optimization_status = optimize_model(
            self.__model__, email, self.cet, solver)

    def __objective_rule(self):
        """Return the proper objective function"""

        def objective_rule(model):
            return sum(model.epsilon[i] ** 2 for i in model.I)

        return objective_rule

    def __regression_rule(self):
        """Return the proper regression constraint"""
        def regression_rule(model, i):
            return self.y[i] == model.alpha[i] + model.beta[i] * self.x[i] + model.epsilon[i]

        return regression_rule

    def __afriat_rule1(self):
        """Return the proper afriat inequality constraint: beta"""
        def afriat_rule1(model, i):
            if i == 0:
                return Constraint.Skip
            if self.x[i] == self.x[self.__model__.I.prevw(i)]:
                return model.beta[i] == model.beta[self.__model__.I.prevw(i)]
            else:
                return model.beta[i] <= model.beta[self.__model__.I.prevw(i)]

        return afriat_rule1

    def __afriat_rule2(self):
        """Return the proper afriat inequality constraint: alpha"""
        def afriat_rule2(model, i):
            if i == 0:
                return Constraint.Skip
            if self.x[i] == self.x[self.__model__.I.prevw(i)]:
                return model.alpha[i] == model.alpha[self.__model__.I.prevw(i)]
            else:
                return model.alpha[i] >= model.alpha[self.__model__.I.prevw(i)]

        return afriat_rule2

    def get_alpha(self):
        """Return alpha value by array"""
        if self.optimization_status == 0:
            self.optimize()
        alpha = list(self.__model__.alpha[:].value)
        return np.asarray(alpha)

    def get_beta(self):
        """Return beta value by array"""
        if self.optimization_status == 0:
            self.optimize()
        beta = list(self.__model__.beta[:].value)
        return np.asarray(beta)
