# import dependencies
from pyomo.environ import ConcreteModel, Set, Var, Objective, minimize, Constraint
from pyomo.core.expr.numvalue import NumericValue
import numpy as np
import pandas as pd

from .constant import CET_ADDI, FUN_PROD, FUN_COST, OPT_DEFAULT, OPT_LOCAL
from .utils import tools


class CSVR:
    """Convex Support Vector Regression (CSVR)
    """

    def __init__(self, y, x, fun=FUN_PROD, epsilon=0.01, C=2):
        """CSVR model

        Args:
            y (float): output variable. 
            x (float): input variables.
            epsilon (float): epsilon-loss function.
            C (float): Regularization parameter.
            fun (String, optional): FUN_PROD (production frontier) or FUN_COST (cost frontier). Defaults to FUN_PROD.
        """
        # TODO(error/warning handling): Check the configuration of the model exist
        self.y, self.x, self.z = tools.assert_valid_basic_data(y, x, z=None)
        self.epsilon = epsilon
        self.C = C
        self.cet = CET_ADDI
        self.fun = fun

        # Initialize the CNLS model
        self.__model__ = ConcreteModel()

        # Initialize the sets
        self.__model__.I = Set(initialize=range(len(self.y)))
        self.__model__.J = Set(initialize=range(len(self.x[0])))

        # Initialize the variables
        self.__model__.alpha = Var(self.__model__.I, doc='alpha')
        self.__model__.beta = Var(self.__model__.I,
                                  self.__model__.J,
                                  doc='beta')
        self.__model__.ksia = Var(self.__model__.I, 
                                  bounds=(0.0, None), 
                                  doc='Ksi a')
        self.__model__.ksib = Var(self.__model__.I, 
                                  bounds=(0.0, None), 
                                  doc='Ksi b')

        # Setup the objective function and constraints
        self.__model__.objective = Objective(rule=self.__objective_rule(),
                                             sense=minimize,
                                             doc='objective function')

        self.__model__.regression_rule1 = Constraint(self.__model__.I,
                                                    rule=self.__regression_rule1(),
                                                    doc='regression equation')

        self.__model__.regression_rule2 = Constraint(self.__model__.I,
                                                    rule=self.__regression_rule2(),
                                                    doc='regression equation')

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
            self.__model__, email, self.cet, solver)

    def __objective_rule(self):
        """Return the proper objective function"""

        def objective_rule(model):
            return self.C * (sum(model.ksia[i] for i in model.I) + sum(model.ksib[i] for i in model.I)) + \
                        sum(model.beta[i, j]**2 for i in model.I for j in model.J)

        return objective_rule

    def __regression_rule1(self):
        """Return the proper regression constraint"""

        def regression_rule1(model, i):
            return self.y[i] - sum(model.beta[i, j] * self.x[i][j] for j in model.J) - model.alpha[i]\
                <= self.epsilon + model.ksia[i]

        return regression_rule1

    def __regression_rule2(self):
        """Return the proper regression constraint"""

        def regression_rule2(model, i):
            return sum(model.beta[i, j] * self.x[i][j] for j in model.J) + model.alpha[i] - self.y[i]\
                <= self.epsilon + model.ksib[i]

        return regression_rule2

    def __afriat_rule(self):
        """Return the proper afriat inequality constraint"""
        if self.fun == FUN_PROD:
            __operator = NumericValue.__le__
        elif self.fun == FUN_COST:
            __operator = NumericValue.__ge__

        def afriat_rule(model, i, h):
            if i == h:
                return Constraint.Skip
            return __operator(
                model.alpha[i] + sum(model.beta[i, j] * self.x[i][j]
                                    for j in model.J),
                model.alpha[h] + sum(model.beta[h, j] * self.x[i][j]
                                    for j in model.J))

        return afriat_rule

    def display_status(self):
        """Display the status of problem"""
        tools.assert_optimized(self.optimization_status)
        print(self.display_status)

    def display_alpha(self):
        """Display alpha value"""
        tools.assert_optimized(self.optimization_status)
        self.__model__.alpha.display()

    def display_beta(self):
        """Display beta value"""
        tools.assert_optimized(self.optimization_status)
        self.__model__.beta.display()

    def get_status(self):
        """Return status"""
        return self.optimization_status

    def get_alpha(self):
        """Return alpha value by array"""
        tools.assert_optimized(self.optimization_status)
        alpha = list(self.__model__.alpha[:].value)
        return np.asarray(alpha)

    def get_beta(self):
        """Return beta value by array"""
        tools.assert_optimized(self.optimization_status)
        beta = np.asarray([i + tuple([j]) for i, j in zip(list(self.__model__.beta),
                                                          list(self.__model__.beta[:, :].value))])
        beta = pd.DataFrame(beta, columns=['Name', 'Key', 'Value'])
        beta = beta.pivot(index='Name', columns='Key', values='Value')
        return beta.to_numpy()
