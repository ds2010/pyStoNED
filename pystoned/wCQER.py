# import dependencies
from pyomo.environ import Objective, minimize
from . import CQER
from .constant import CET_ADDI, FUN_PROD, RTS_VRS
from .utils import tools


class wCQR(CQER.CQR):
    """Weighted Convex Quantile Regression (wCQR)
    """

    def __init__(self, y, x, w, tau, z=None, cet=CET_ADDI, fun=FUN_PROD, rts=RTS_VRS):
        """wCQR model

        Args:
            y (float): output variable. 
            x (float): input variables.
            w (float): weight variable.
            z (float, optional): Contextual variable(s). Defaults to None.
            tau (float): quantile.
            cet (String, optional): CET_ADDI (additive composite error term) or CET_MULT (multiplicative composite error term). Defaults to CET_ADDI.
            fun (String, optional): FUN_PROD (production frontier) or FUN_COST (cost frontier). Defaults to FUN_PROD.
            rts (String, optional): RTS_VRS (variable returns to scale) or RTS_CRS (constant returns to scale). Defaults to RTS_VRS.
        """
        # TODO(error/warning handling): Check the configuration of the model exist
        super().__init__(y, x, tau, z, cet, fun, rts)
        self.w = tools.trans_list(tools.to_1d_list(w))

        self.__model__.objective.deactivate()
        self.__model__.weighted_objective = Objective(
            rule=self.__weighted_objective_rule(), sense=minimize, doc='weighted objective rule')

    def __weighted_objective_rule(self):
        def weighted_objective_rule(model):
            return self.tau * sum(self.w[i] * model.epsilon_plus[i] for i in model.I) \
                + (1 - self.tau) * \
                sum(self.w[i] * model.epsilon_minus[i] for i in model.I)

        return weighted_objective_rule


class wCER(CQER.CQR):
    """Weighted Convex Expectile Regression (wCER)
    """

    def __init__(self, y, x, w, tau, z=None, cet=CET_ADDI, fun=FUN_PROD, rts=RTS_VRS):
        """wCER model

        Args:
            y (float): output variable. 
            x (float): input variables.
            w (float): weight variable.
            z (float, optional): Contextual variable(s). Defaults to None.
            tau (float): expectile.
            cet (String, optional): CET_ADDI (additive composite error term) or CET_MULT (multiplicative composite error term). Defaults to CET_ADDI.
            fun (String, optional): FUN_PROD (production frontier) or FUN_COST (cost frontier). Defaults to FUN_PROD.
            rts (String, optional): RTS_VRS (variable returns to scale) or RTS_CRS (constant returns to scale). Defaults to RTS_VRS.
        """
        super().__init__(y, x, tau, z, cet, fun, rts)
        self.w = tools.trans_list(tools.to_1d_list(w))

        self.__model__.objective.deactivate()
        self.__model__.weighted_squared_objective = Objective(
            rule=self.__weighted_squared_objective_rule(), sense=minimize, doc='weighted square objective rule')

    def __weighted_squared_objective_rule(self):
        def weighted_squared_objective_rule(model):
            return self.tau * sum(self.w[i] * model.epsilon_plus[i] ** 2 for i in model.I) \
                + (1 - self.tau) * \
                sum(self.w[i] * model.epsilon_minus[i] ** 2 for i in model.I)

        return weighted_squared_objective_rule
