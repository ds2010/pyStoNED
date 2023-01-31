# import dependencies
from pyomo.environ import Objective, minimize
from . import CNLS
from .constant import CET_ADDI, FUN_PROD, RTS_VRS


class pCNLS(CNLS.CNLS):
    """Convex Nonparametric Least Square with squared L2-norm regularization (pCNLS)
    """

    def __init__(self, y, x, eta, z=None, cet=CET_ADDI, fun=FUN_PROD, rts=RTS_VRS):
        """pCNLS model

        Args:
            y (float): output variable. 
            x (float): input variables.
            eta (float): regularization parameter.
            z (float, optional): Contextual variable(s). Defaults to None.
            cet (String, optional): CET_ADDI (additive composite error term) or CET_MULT (multiplicative composite error term). Defaults to CET_ADDI.
            fun (String, optional): FUN_PROD (production frontier) or FUN_COST (cost frontier). Defaults to FUN_PROD.
            rts (String, optional): RTS_VRS (variable returns to scale) or RTS_CRS (constant returns to scale). Defaults to RTS_VRS.
        """
        self.eta = eta
        CNLS.CNLS.__init__(self, y, x, z, cet, fun, rts)
        self.__model__.objective.deactivate()
        self.__model__.new_objective = Objective(rule=self.__new_objective_rule(),
                                             sense=minimize,
                                             doc='objective function')
    
    def __new_objective_rule(self):
        """Return the proper objective function"""

        def objective_rule(model):
            return sum(model.epsilon[i] ** 2 for i in model.I) \
                   + self.eta * sum(model.beta[ij] ** 2 for ij in model.I * model.J)

        return objective_rule
