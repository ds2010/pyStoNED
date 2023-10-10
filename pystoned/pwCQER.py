# import dependencies
from pyomo.environ import Objective, minimize, Constraint
from . import wCQER
from .constant import CET_ADDI, FUN_PROD, RTS_VRS


class pwCQR(wCQER.wCQR):
    """penalized Weighted Convex Quantile Regression (pwCQR)
    """

    def __init__(self, y, x, w, tau, eta, z=None, cet=CET_ADDI, fun=FUN_PROD, rts=RTS_VRS, penalty=1):
        """pwCQR model

        Args:
            y (float): output variable. 
            x (float): input variables.
            w (float): weight variable.
            eta (float): tuning parameter.
            z (float, optional): Contextual variable(s). Defaults to None.
            tau (float): quantile.
            cet (String, optional): CET_ADDI (additive composite error term) or CET_MULT (multiplicative composite error term). Defaults to CET_ADDI.
            fun (String, optional): FUN_PROD (production frontier) or FUN_COST (cost frontier). Defaults to FUN_PROD.
            rts (String, optional): RTS_VRS (variable returns to scale) or RTS_CRS (constant returns to scale). Defaults to RTS_VRS.
            penalty (int, optional): penalty=1 (L1 norm), penalty=2 (L2 norm), and penalty=3 (Lipschitz norm). Defaults to 1.
        """
        # TODO(error/warning handling): Check the configuration of the model exist
        self.eta = eta
        wCQER.wCQR.__init__(self, y, x, w, tau, z, cet, fun, rts)
        if penalty == 1 or penalty == 2:
            self.__model__.weighted_objective.deactivate()

        if penalty == 1:
            self.__model__.new_objective = Objective(rule=self.__new_objective_rule(),
                                                     sense=minimize,
                                                     doc='objective function')
        elif penalty == 2:
            self.__model__.new_objective = Objective(rule=self.__new_objective_rule2(),
                                                     sense=minimize,
                                                     doc='objective function')
        elif penalty == 3:
            self.__model__.lipschitz_norm = Constraint(self.__model__.I,
                                                       rule=self.__lipschitz_rule(),
                                                       doc='Lipschitz norm')
        else:
            raise ValueError('Penalty must be 1, 2, or 3.')

    def __new_objective_rule(self):
        """Return the proper objective function"""

        def objective_rule(model):
            return self.tau * sum(self.w[i] * model.epsilon_plus[i] for i in model.I) \
                + (1 - self.tau) * sum(self.w[i] * model.epsilon_minus[i] for i in model.I) \
                + self.eta * sum(model.beta[ij] for ij in model.I * model.J)

        return objective_rule

    def __new_objective_rule2(self):
        """Return the proper objective function"""

        def objective_rule(model):
            return self.tau * sum(self.w[i] * model.epsilon_plus[i] for i in model.I) \
                + (1 - self.tau) * sum(self.w[i] * model.epsilon_minus[i] for i in model.I) \
                + self.eta * sum(model.beta[ij] **
                                 2 for ij in model.I * model.J)

        return objective_rule

    def __lipschitz_rule(self):
        """Lipschitz norm"""

        def lipschitz_rule(model, i):
            return sum(model.beta[i, j] ** 2 for j in model.J) <= self.eta**2

        return lipschitz_rule


class pwCER(wCQER.wCER):
    """penalized Weighted Convex Expectile Regression (pwCER)
    """

    def __init__(self, y, x, w, tau, eta, z=None, cet=CET_ADDI, fun=FUN_PROD, rts=RTS_VRS, penalty=1):
        """pwCER model

        Args:
            y (float): output variable. 
            x (float): input variables.
            w (float): weight variable.
            eta (float): tuning parameter.
            z (float, optional): Contextual variable(s). Defaults to None.
            tau (float): expectile.
            cet (String, optional): CET_ADDI (additive composite error term) or CET_MULT (multiplicative composite error term). Defaults to CET_ADDI.
            fun (String, optional): FUN_PROD (production frontier) or FUN_COST (cost frontier). Defaults to FUN_PROD.
            rts (String, optional): RTS_VRS (variable returns to scale) or RTS_CRS (constant returns to scale). Defaults to RTS_VRS.
            penalty (int, optional): penalty=1 (L1 norm), penalty=2 (L2 norm), and penalty=3 (Lipschitz norm). Defaults to 1.
        """
        self.eta = eta
        wCQER.wCER.__init__(self, y, x, w, tau, z, cet, fun, rts)
        if penalty == 1 or penalty == 2:
            self.__model__.weighted_squared_objective.deactivate()

        if penalty == 1:
            self.__model__.new_objective = Objective(rule=self.__new_squared_objective_rule(),
                                                     sense=minimize,
                                                     doc='objective function')
        elif penalty == 2:
            self.__model__.new_objective = Objective(rule=self.__new_squared_objective_rule2(),
                                                     sense=minimize,
                                                     doc='objective function')
        elif penalty == 3:
            self.__model__.lipschitz_norm = Constraint(self.__model__.I,
                                                       rule=self.__lipschitz_rule(),
                                                       doc='Lipschitz norm')
        else:
            raise ValueError('Penalty must be 1, 2, or 3.')

    def __new_squared_objective_rule(self):
        """Return the proper objective function"""

        def objective_rule(model):
            return self.tau * sum(self.w[i] * model.epsilon_plus[i] ** 2 for i in model.I) \
                + (1 - self.tau) * sum(self.w[i] * model.epsilon_minus[i] ** 2 for i in model.I) \
                + self.eta * sum(model.beta[ij] for ij in model.I * model.J)

        return objective_rule

    def __new_squared_objective_rule2(self):
        """Return the proper objective function"""

        def objective_rule(model):
            return self.tau * sum(self.w[i] * model.epsilon_plus[i] ** 2 for i in model.I) \
                + (1 - self.tau) * sum(self.w[i] * model.epsilon_minus[i] ** 2 for i in model.I) \
                + self.eta * sum(model.beta[ij] **
                                 2 for ij in model.I * model.J)

        return objective_rule

    def __lipschitz_rule(self):
        """Lipschitz norm"""

        def lipschitz_rule(model, i):
            return sum(model.beta[i, j] ** 2 for j in model.J) <= self.eta**2

        return lipschitz_rule
