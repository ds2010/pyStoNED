# import dependencies
from pyomo.environ import ConcreteModel, Set, Var, Objective, minimize, Constraint
import numpy as np

from .constant import CET_ADDI, OPT_LOCAL, OPT_DEFAULT
from .utils import tools


class uCQR:
    """univariate Convex quantile regression (CQR)
    """

    def __init__(self, y, x, tau):
        """uCQR model

        Args:
            y (float): output variable. 
            x (float): input variable.
            tau (float): quantile.
        """
        # TODO(error/warning handling): Check the configuration of the model exist
        self.y, self.x = tools.assert_valid_univariate_data(y, x)
        self.tau = tau
        self.cet = CET_ADDI

        # Initialize the uCQR model
        self.__model__ = ConcreteModel()

        # Initialize the sets
        self.__model__.I = Set(initialize=range(len(self.y)))

        # Initialize the variables
        self.__model__.alpha = Var(self.__model__.I, doc='alpha')
        self.__model__.beta = Var(self.__model__.I,
                                  bounds=(0.0, None),
                                  doc='beta')
        self.__model__.epsilon = Var(self.__model__.I, doc='error term')
        self.__model__.epsilon_plus = Var(
            self.__model__.I, bounds=(0.0, None), doc='positive error term')
        self.__model__.epsilon_minus = Var(
            self.__model__.I, bounds=(0.0, None), doc='negative error term')
        self.__model__.frontier = Var(self.__model__.I,
                                      bounds=(0.0, None),
                                      doc='estimated frontier')

        # Setup the objective function and constraints
        self.__model__.objective = Objective(rule=self.__objective_rule(),
                                             sense=minimize,
                                             doc='objective function')

        self.__model__.error_decomposition = Constraint(self.__model__.I,
                                                        rule=self.__error_decomposition(),
                                                        doc='decompose error term')

        self.__model__.regression_rule = Constraint(self.__model__.I,
                                                    rule=self.__regression_rule(),
                                                    doc='regression equation')

        self.__model__.afriat_rule1 = Constraint(self.__model__.I,
                                                 rule=self.__afriat_rule1(),
                                                 doc='afriat inequality')

        self.__model__.afriat_rule2 = Constraint(self.__model__.I,
                                                 rule=self.__afriat_rule2(),
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
            return self.tau * sum(model.epsilon_plus[i] for i in model.I) \
                   + (1 - self.tau) * sum(model.epsilon_minus[i] for i in model.I)

        return objective_rule

    def __error_decomposition(self):
        """Return the constraint decomposing error to positive and negative terms"""

        def error_decompose_rule(model, i):
            return model.epsilon[i] == model.epsilon_plus[i] - model.epsilon_minus[i]

        return error_decompose_rule

    def __regression_rule(self):
        """Return the proper regression constraint"""

        def regression_rule(model, i):
            return self.y[i] == model.alpha[i] + model.beta[i] * self.x[i] + model.epsilon[i]

        return regression_rule

    def __afriat_rule1(self):
        """Return the proper afriat inequality constraint: beta"""

        def afriat_rule1(model, i):
            if self.x[i] == self.x[self.__model__.I.prevw(i)]:
                return model.beta[i] == model.beta[self.__model__.I.prevw(i)]
            else:
                return model.beta[i] <= model.beta[self.__model__.I.prevw(i)]

        return afriat_rule1

    def __afriat_rule2(self):
        """Return the proper afriat inequality constraint: alpha"""

        def afriat_rule2(model, i):
            if self.x[i] == self.x[self.__model__.I.prevw(i)]:
                return model.alpha[i] == model.alpha[self.__model__.I.prevw(i)]
            else:
                return model.alpha[i] >= model.alpha[self.__model__.I.prevw(i)]

        return afriat_rule2

    def display_status(self):
        """Display the status of problem"""
        print(self.optimization_status)

    def display_alpha(self):
        """Display alpha value"""
        tools.assert_optimized(self.optimization_status)
        self.__model__.alpha.display()

    def display_beta(self):
        """Display beta value"""
        tools.assert_optimized(self.optimization_status)
        self.__model__.beta.display()

    def display_residual(self):
        """Dispaly residual value"""
        tools.assert_optimized(self.optimization_status)
        self.__model__.epsilon.display()

    def display_positive_residual(self):
        """Dispaly positive residual value"""
        tools.assert_optimized(self.optimization_status)
        self.__model__.epsilon_plus.display()

    def display_negative_residual(self):
        """Dispaly negative residual value"""
        tools.assert_optimized(self.optimization_status)
        self.__model__.epsilon_minus.display()

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
        beta = list(self.__model__.beta[:].value)
        return np.asarray(beta)

    def get_residual(self):
        """Return residual value by array"""
        tools.assert_optimized(self.optimization_status)
        residual = list(self.__model__.epsilon[:].value)
        return np.asarray(residual)

    def get_positive_residual(self):
        """Return positive residual value by array"""
        tools.assert_optimized(self.optimization_status)
        residual_plus = list(self.__model__.epsilon_plus[:].value)
        return np.asarray(residual_plus)

    def get_negative_residual(self):
        """Return negative residual value by array"""
        tools.assert_optimized(self.optimization_status)
        residual_minus = list(self.__model__.epsilon_minus[:].value)
        return np.asarray(residual_minus)

    def get_frontier(self):
        """Return estimated frontier value by array"""
        tools.assert_optimized(self.optimization_status)
        frontier = np.asarray(self.y) - self.get_residual()
        return np.asarray(frontier)


class uCER(uCQR):
    """univariate Convex expectile regression (uCER)
    """

    def __init__(self, y, x, tau):
        """uCER model

        Args:
            y (float): output variable. 
            x (float): input variable.
            tau (float): expectile.
        """
        super().__init__(y, x, tau)
        self.__model__.objective.deactivate()
        self.__model__.squared_objective = Objective(
            rule=self.__squared_objective_rule(), sense=minimize, doc='squared objective rule')

    def __squared_objective_rule(self):
        def squared_objective_rule(model):
            return self.tau * sum(model.epsilon_plus[i] ** 2 for i in model.I) \
                   + (1 - self.tau) * \
                   sum(model.epsilon_minus[i] ** 2 for i in model.I)

        return squared_objective_rule
