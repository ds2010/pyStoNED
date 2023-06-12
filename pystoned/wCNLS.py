# import dependencies
from pyomo.environ import Objective, minimize
from . import CNLS
from .constant import CET_ADDI, FUN_PROD, RTS_VRS
from .utils import tools


class wCNLS(CNLS.CNLS):
    """Weighted Convex Nonparametric Least Square (wCNLS)
    """

    def __init__(self, y, x, w, z=None, cet=CET_ADDI, fun=FUN_PROD, rts=RTS_VRS):
        """wCNLS model

        Args:
            y (float): output variable. 
            x (float): input variables.
            w (float): weight variable.
            z (float, optional): Contextual variable(s). Defaults to None.
            cet (String, optional): CET_ADDI (additive composite error term) or CET_MULT (multiplicative composite error term). Defaults to CET_ADDI.
            fun (String, optional): FUN_PROD (production frontier) or FUN_COST (cost frontier). Defaults to FUN_PROD.
            rts (String, optional): RTS_VRS (variable returns to scale) or RTS_CRS (constant returns to scale). Defaults to RTS_VRS.
        """
        # TODO(error/warning handling): Check the configuration of the model exist
        super().__init__(y, x, z, cet, fun, rts)
        self.w = tools.trans_list(tools.to_1d_list(w))

        self.__model__.objective.deactivate()
        self.__model__.weighted_objective = Objective(
            rule=self.__weighted_objective_rule(), sense=minimize, doc='weighted objective rule')

    def __weighted_objective_rule(self):
        def weighted_objective_rule(model):
            return sum(self.w[i] * model.epsilon[i] ** 2 for i in model.I)

        return weighted_objective_rule
