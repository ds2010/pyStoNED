# import dependencies
from pyomo.environ import ConcreteModel, Set, Var, Objective, minimize, Constraint
import numpy as np
import pandas as pd

from .constant import CET_ADDI,  OPT_LOCAL, OPT_DEFAULT
from .utils import tools


class sCQR:
    """Simultaneous estimation of CQR
    """

    def __init__(self, y, x, tau, C):
        """sCQR model

        Args:
            y (float): output variable. 
            x (float): input variables.
            tau: vector of quantile.
            C: interval (small positive value)
        """
        # TODO(error/warning handling): Check the configuration of the model exist
        self.y, self.x = tools.to_1d_list(tools.trans_list(y)), tools.to_2d_list(tools.trans_list(x))
        self.tau = tools.to_1d_list(tools.trans_list(tau))
        self.C = C
        self.cet = CET_ADDI

        # Initialize the CQR model
        self.__model__ = ConcreteModel()

        # Initialize the sets
        self.__model__.I = Set(initialize=range(len(self.y)))
        self.__model__.J = Set(initialize=range(len(self.x[0])))
        self.__model__.K = Set(initialize=range(len(self.tau)))
        self.__model__.W = Set(initialize=range(len(self.tau)-1))

        # Initialize the variables
        self.__model__.alpha = Var(self.__model__.K, self.__model__.I, doc='alpha')
        self.__model__.beta = Var(self.__model__.K, self.__model__.I, self.__model__.J, bounds=(0.0, None), doc='beta')
        self.__model__.epsilon_plus = Var(self.__model__.K, self.__model__.I, bounds=(0.0, None), doc='positive error term')
        self.__model__.epsilon_minus = Var(self.__model__.K, self.__model__.I, bounds=(0.0, None), doc='negative error term')
        
        # Setup the objective function and constraints
        self.__model__.objective = Objective(rule=self.__objective_rule(), sense=minimize, doc='objective function')

        self.__model__.regression_rule = Constraint(self.__model__.K, self.__model__.I, rule=self.__regression_rule(),
                                                     doc='regression equation')
        
        self.__model__.afriat_rule = Constraint(self.__model__.K, self.__model__.I, self.__model__.I, rule=self.__afriat_rule(),
                                                 doc='afriat inequality')

        self.__model__.noncrossing_rule = Constraint(self.__model__.W, self.__model__.I, rule=self.__noncrossing_rule(),
                                                     doc='non-crossing constraint')

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
            return sum(self.tau[k] * sum(model.epsilon_plus[k, i] for i in model.I) \
                   + (1 - self.tau[k]) * sum(model.epsilon_minus[k, i] for i in model.I) for k in model.K)

        return objective_rule

    def __regression_rule(self):
        """Return the proper regression constraint"""

        def regression_rule(model, k, i):
            return self.y[i] == model.alpha[k, i] + sum(model.beta[k, i, j] * self.x[i][j] for j in model.J) + \
                   model.epsilon_plus[k, i] - model.epsilon_minus[k, i]

        return regression_rule

    def __afriat_rule(self):
        """Return the proper afriat inequality constraint"""

        def afriat_rule(model, k, i, h):
            if i == h:
                return Constraint.Skip
            return model.alpha[k, i] + sum(model.beta[k, i, j] * self.x[i][j] for j in model.J) \
                    <= model.alpha[k, h] + sum(model.beta[k, h, j] * self.x[i][j] for j in model.J)

        return afriat_rule

    def __noncrossing_rule(self):
        """Return the proper non-crossing constraint"""

        def noncrossing_rule(model, w, i):
            return model.alpha[w, i] + sum(model.beta[w, i, j] * self.x[i][j] for j in model.J) + self.C <= \
                        model.alpha[w+1, i] + sum(model.beta[w+1, i, j] * self.x[i][j] for j in model.J) 

        return noncrossing_rule

    def get_alpha(self):
        """Return alpha value by array"""
        tools.assert_optimized(self.optimization_status)
        alpha = np.asarray([i + tuple([j]) for i, j in zip(list(self.__model__.alpha),
                                                          list(self.__model__.alpha[:, :].value))])
        alpha = pd.DataFrame(alpha, columns=['Name', 'Key', 'Value'])
        alpha = alpha.pivot(index='Name', columns='Key', values='Value')
        return alpha.to_numpy().T

    def get_beta(self):
        """Return beta value by array"""
        tools.assert_optimized(self.optimization_status)
        beta = np.asarray([i + tuple([j]) for i, j in zip(list(self.__model__.beta),
                                                          list(self.__model__.beta[:, :, :].value))])
        beta = pd.DataFrame(beta, columns=['tau', 'n', 'd', 'beta'])
        betanew = []
        for i in range(len(self.tau)):
            beta_new = beta.loc[beta['tau'] == i]
            del beta_new["tau"]
            betanew.append(beta_new.pivot(index='n', columns='d', values='beta').to_numpy())
        return np.array(betanew).reshape(len(self.tau)*len(self.y), len(self.x[0]))

    def get_positive_residual(self):
        """Return positive residual value by array"""
        tools.assert_optimized(self.optimization_status)
        residual_plus = np.asarray([i + tuple([j]) for i, j in zip(list(self.__model__.epsilon_plus),
                                                          list(self.__model__.epsilon_plus[:, :].value))])
        residual_plus = pd.DataFrame(residual_plus, columns=['Name', 'Key', 'Value'])
        residual_plus = residual_plus.pivot(index='Name', columns='Key', values='Value')
        return residual_plus.to_numpy().T

    def get_negative_residual(self):
        """Return negative residual value by array"""
        tools.assert_optimized(self.optimization_status)
        residual_minus = np.asarray([i + tuple([j]) for i, j in zip(list(self.__model__.epsilon_minus),
                                                          list(self.__model__.epsilon_minus[:, :].value))])
        residual_minus = pd.DataFrame(residual_minus, columns=['Name', 'Key', 'Value'])
        residual_minus = residual_minus.pivot(index='Name', columns='Key', values='Value')
        return residual_minus.to_numpy().T

    def get_frontier(self):
        """Return estimated frontier value by array"""
        tools.assert_optimized(self.optimization_status)
        frontier = np.tile(np.asarray(self.y).reshape(len(self.y), 1), len(self.tau)) \
                    - (self.get_positive_residual() - self.get_negative_residual()) 
        return np.asarray(frontier)       


class sCER(sCQR):
    """Simultaneous estimation of CER
    """

    def __init__(self, y, x, tau, C):
        """sCER model

        Args:
            y (float): output variable. 
            x (float): input variables.
            tau: vector of expectile.
            C: interval (small positive value)
        """
        super().__init__(y, x, tau, C)
        self.__model__.objective.deactivate()
        self.__model__.squared_objective = Objective(
            rule=self.__squared_objective_rule(), sense=minimize, doc='squared objective rule')

    def __squared_objective_rule(self):
        def squared_objective_rule(model):
            return sum(self.tau[k] * sum(model.epsilon_plus[k, i] **2 for i in model.I) \
                   + (1 - self.tau[k]) * sum(model.epsilon_minus[k, i] **2 for i in model.I) for k in model.K)

        return squared_objective_rule