# import dependencies
from pyomo.environ import Objective, minimize
from . import pCQER
from .constant import CET_ADDI, FUN_PROD, RTS_VRS
from .utils import tools


class pwCQR(pCQER.pCQR):
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
            penalty (int, optional): penalty=1 (L1 norm) and penalty=2 (L2 norm). Defaults to 1.
        """
        # TODO(error/warning handling): Check the configuration of the model exist
        self.w = tools.trans_list(tools.to_1d_list(w))
        pCQER.pCQR.__init__(self, y, x, tau, eta, z, cet, fun, rts, penalty)

        self.__model__.objective.deactivate()
        self.__model__.new_objective = Objective(rule=self.__new_objective_rule(),
                                                 sense=minimize,
                                                 doc='objective function')

    def __new_objective_rule(self):
        """Return the proper objective function"""

        def objective_rule(model):
            return self.tau * sum(self.w[i] * model.epsilon_plus[i] for i in model.I) \
                + (1 - self.tau) * \
                sum(self.w[i] * model.epsilon_minus[i] for i in model.I)

        return objective_rule


class pwCER(pCQER.pCER):
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
            penalty (int, optional): penalty=1 (L1 norm) and penalty=2 (L2 norm). Defaults to 1.
        """
        self.w = tools.trans_list(tools.to_1d_list(w))
        pCQER.pCER.__init__(self, y, x, tau, eta, z, cet, fun, rts, penalty)

        self.__model__.squared_objective.deactivate()
        self.__model__.new_objective = Objective(rule=self.__new_squared_objective_rule(),
                                                 sense=minimize,
                                                 doc='objective function')

    def __new_squared_objective_rule(self):
        """Return the proper objective function"""

        def objective_rule(model):
            return self.tau * sum(self.w[i] * model.epsilon_plus[i] ** 2 for i in model.I) \
                + (1 - self.tau) * \
                sum(self.w[i] * model.epsilon_minus[i] ** 2 for i in model.I)

        return objective_rule
