# import dependencies
from pyomo.environ import Objective, minimize

from . import CQER
from .constant import CET_ADDI, FUN_PROD, RTS_VRS
from .utils import tools


class pwCQR(CQER.CQR):
    """penalized Weighted Convex Quantile Regression (pwCQR)
    """

    def __init__(self, y, x, w, eta, tau, z=None, cet=CET_ADDI, fun=FUN_PROD, rts=RTS_VRS, penalty=1):
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
            penalty (int, optional): penalty=1 (L1 norm) and penalty=2 (L2 norm). Defaults to 1.
        """
        # TODO(error/warning handling): Check the configuration of the model exist
        self.eta = eta
        self.w = tools.trans_list(tools.to_1d_list(w))
        CQER.CQR.__init__(self, y, x, tau, z, cet, fun, rts)
        self.__model__.objective.deactivate()
        if penalty == 1:
            self.__model__.new_objective = Objective(rule=self.__new_objective_rule(),
                                                     sense=minimize,
                                                     doc='objective function')
        elif penalty == 2:
            self.__model__.new_objective = Objective(rule=self.__new_objective_rule2(),
                                                     sense=minimize,
                                                     doc='objective function')
        else:
            raise ValueError('Penalty must be 1 or 2.')

    def __new_objective_rule(self):
        """Return the proper objective function"""

        def objective_rule(model):
            return self.tau * sum(model.epsilon_plus[i] for i in model.I) \
                + (1 - self.tau) * sum(model.epsilon_minus[i] for i in model.I) \
                + self.eta * sum(abs(model.beta[ij])
                                 for ij in model.I * model.J)

        return objective_rule

    def __new_objective_rule2(self):
        """Return the proper objective function"""

        def objective_rule(model):
            return self.tau * sum(model.epsilon_plus[i] for i in model.I) \
                + (1 - self.tau) * sum(model.epsilon_minus[i] for i in model.I) \
                + self.eta * sum(model.beta[ij] **
                                 2 for ij in model.I * model.J)

        return objective_rule


class pwCER(CQER.CER):
    """penalized Weighted Convex Expectile Regression (pwCER)
    """

    def __init__(self, y, x, w, eta, tau, z=None, cet=CET_ADDI, fun=FUN_PROD, rts=RTS_VRS, penalty=1):
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
            penalty (int, optional): penalty=1 (L1 norm) and penalty=2 (L2 norm). Defaults to 1.
        """
        self.eta = eta
        self.w = tools.trans_list(tools.to_1d_list(w))
        CQER.CER.__init__(self, y, x, tau, z, cet, fun, rts)
        self.__model__.squared_objective.deactivate()

        if penalty == 1:
            self.__model__.new_objective = Objective(rule=self.__new_squared_objective_rule(),
                                                     sense=minimize,
                                                     doc='objective function')
        elif penalty == 2:
            self.__model__.new_objective = Objective(rule=self.__new_squared_objective_rule2(),
                                                     sense=minimize,
                                                     doc='objective function')
        else:
            raise ValueError('Penalty must be 1 or 2.')

    def __new_squared_objective_rule(self):
        """Return the proper objective function"""

        def objective_rule(model):
            return self.tau * sum(model.epsilon_plus[i] ** 2 for i in model.I) \
                + (1 - self.tau) * sum(model.epsilon_minus[i] ** 2 for i in model.I) \
                + self.eta * sum(abs(model.beta[ij])
                                 for ij in model.I * model.J)

        return objective_rule

    def __new_squared_objective_rule2(self):
        """Return the proper objective function"""

        def objective_rule(model):
            return self.tau * sum(model.epsilon_plus[i] ** 2 for i in model.I) \
                + (1 - self.tau) * sum(model.epsilon_minus[i] ** 2 for i in model.I) \
                + self.eta * sum(model.beta[ij] **
                                 2 for ij in model.I * model.J)

        return objective_rule
