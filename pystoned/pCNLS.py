# import dependencies
from pyomo.environ import Objective, minimize, Constraint
from . import CNLS
from .constant import CET_ADDI, FUN_PROD, RTS_VRS


class pCNLS(CNLS.CNLS):
    """penalized Convex Nonparametric Least Square (pCNLS)
    """

    def __init__(self, y, x, eta, z=None, cet=CET_ADDI, fun=FUN_PROD, rts=RTS_VRS, penalty=1):
        """pCNLS model

        Args:
            y (float): output variable. 
            x (float): input variables.
            eta (float): regularization parameter.
            z (float, optional): Contextual variable(s). Defaults to None.
            cet (String, optional): CET_ADDI (additive composite error term) or CET_MULT (multiplicative composite error term). Defaults to CET_ADDI.
            fun (String, optional): FUN_PROD (production frontier) or FUN_COST (cost frontier). Defaults to FUN_PROD.
            rts (String, optional): RTS_VRS (variable returns to scale) or RTS_CRS (constant returns to scale). Defaults to RTS_VRS.
            penalty (int, optional): penalty=1 (L1 norm), penalty=2 (L2 norm), and penalty=3 (Lipschitz norm). Defaults to 1.
        """
        self.eta = eta
        CNLS.CNLS.__init__(self, y, x, z, cet, fun, rts)
        if penalty == 1 or penalty == 2:
            self.__model__.objective.deactivate()

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
            return sum(model.epsilon[i] ** 2 for i in model.I) \
                + self.eta * sum(model.beta[ij] for ij in model.I * model.J)

        return objective_rule

    def __new_objective_rule2(self):
        """Return the proper objective function"""

        def objective_rule(model):
            return sum(model.epsilon[i] ** 2 for i in model.I) \
                + self.eta * sum(model.beta[ij] **
                                 2 for ij in model.I * model.J)

        return objective_rule

    def __lipschitz_rule(self):
        """Lipschitz norm"""

        def lipschitz_rule(model, i):
            return sum(model.beta[i, j] ** 2 for j in model.J) <= self.eta**2

        return lipschitz_rule