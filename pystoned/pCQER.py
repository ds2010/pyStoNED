# import dependencies
from pyomo.environ import Objective, minimize
from . import CQER
from .constant import CET_ADDI, FUN_PROD, RTS_VRS


class pCQR(CQER.CQR):
    """Convex quantile regression with squared L2-norm regularization (pCQR)
    """

    def __init__(self, y, x, tau, eta, z=None, cet=CET_ADDI, fun=FUN_PROD, rts=RTS_VRS):
        """pCQR model

        Args:
            y (float): output variable. 
            x (float): input variables.
            z (float, optional): Contextual variable(s). Defaults to None.
            tau (float): quantile.
            eta (float): tuning parameter.
            cet (String, optional): CET_ADDI (additive composite error term) or CET_MULT (multiplicative composite error term). Defaults to CET_ADDI.
            fun (String, optional): FUN_PROD (production frontier) or FUN_COST (cost frontier). Defaults to FUN_PROD.
            rts (String, optional): RTS_VRS (variable returns to scale) or RTS_CRS (constant returns to scale). Defaults to RTS_VRS.
        """
        self.eta = eta
        CQER.CQR.__init__(self, y, x, tau, z, cet, fun, rts)
        self.__model__.objective.deactivate()
        self.__model__.new_objective = Objective(rule=self.__new_objective_rule(),
                                             sense=minimize,
                                             doc='objective function')
    
    def __new_objective_rule(self):
        """Return the proper objective function"""

        def objective_rule(model):
            return self.tau * sum(model.epsilon_plus[i] for i in model.I) \
                   + (1 - self.tau) * sum(model.epsilon_minus[i] for i in model.I) \
                   + self.eta * sum(model.beta[ij] ** 2 for ij in model.I * model.J)

        return objective_rule


class pCER(CQER.CER):
    """Convex expectile regression with squared L2-norm regularization (pCER)
    """

    def __init__(self, y, x, tau, eta, z=None, cet=CET_ADDI, fun=FUN_PROD, rts=RTS_VRS):
        """pCER model

        Args:
            y (float): output variable. 
            x (float): input variables.
            z (float, optional): Contextual variable(s). Defaults to None.
            tau (float): expectile.
            eta (float): tuning parameter.
            cet (String, optional): CET_ADDI (additive composite error term) or CET_MULT (multiplicative composite error term). Defaults to CET_ADDI.
            fun (String, optional): FUN_PROD (production frontier) or FUN_COST (cost frontier). Defaults to FUN_PROD.
            rts (String, optional): RTS_VRS (variable returns to scale) or RTS_CRS (constant returns to scale). Defaults to RTS_VRS.
        """
        self.eta = eta
        CQER.CER.__init__(self, y, x, tau, z, cet, fun, rts)
        self.__model__.squared_objective.deactivate()
        self.__model__.new_squared_objective = Objective(rule=self.__new_squared_objective_rule(),
                                             sense=minimize,
                                             doc='objective function')
    
    def __new_squared_objective_rule(self):
        """Return the proper objective function"""

        def objective_rule(model):
            return self.tau * sum(model.epsilon_plus[i] ** 2 for i in model.I) \
                   + (1 - self.tau) * sum(model.epsilon_minus[i] ** 2 for i in model.I) \
                   + self.eta * sum(model.beta[ij] ** 2 for ij in model.I * model.J)

        return objective_rule
