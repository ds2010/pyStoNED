# import dependencies
from pyomo.environ import Objective, minimize
from . import pCNLS
from .constant import CET_ADDI, FUN_PROD, RTS_VRS
from .utils import tools


class pwCNLS(pCNLS.pCNLS):
    """penalized Weighted Convex Nonparametric Least Square (pwCNLS)
    """

    def __init__(self, y, x, w, eta, z=None, cet=CET_ADDI, fun=FUN_PROD, rts=RTS_VRS, penalty=1):
        """wCNLS model

        Args:
            y (float): output variable. 
            x (float): input variables.
            w (float): weight variable.
            eta (float): regularization parameter.
            z (float, optional): Contextual variable(s). Defaults to None.
            cet (String, optional): CET_ADDI (additive composite error term) or CET_MULT (multiplicative composite error term). Defaults to CET_ADDI.
            fun (String, optional): FUN_PROD (production frontier) or FUN_COST (cost frontier). Defaults to FUN_PROD.
            rts (String, optional): RTS_VRS (variable returns to scale) or RTS_CRS (constant returns to scale). Defaults to RTS_VRS.
            penalty (int, optional): penalty=1 (L1 norm) and penalty=2 (L2 norm). Defaults to 1.
        """
        # TODO(error/warning handling): Check the configuration of the model exist
        self.w = tools.trans_list(tools.to_1d_list(w))
        pCNLS.pCNLS.__init__(self, y, x, eta, z, cet, fun, rts, penalty)

        self.__model__.objective.deactivate()
        self.__model__.new_objective = Objective(rule=self.__new_objective_rule(),
                                                 sense=minimize,
                                                 doc='weighted objective function')

    def __new_objective_rule(self):
        """Return the proper objective function"""

        def objective_rule(model):
            return sum(self.w[i] * model.epsilon[i] ** 2 for i in model.I)

        return objective_rule
