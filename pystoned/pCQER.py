# import dependencies
from pyomo.environ import Constraint, Var
from . import CQER
from .constant import CET_ADDI, FUN_PROD, RTS_VRS


class pCQR(CQER.CQR):
    """penalized convex quantile regression (pCQR)
    """

    def __init__(self, y, x, tau, eta, z=None, cet=CET_ADDI, fun=FUN_PROD, rts=RTS_VRS, penalty=1):
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
            penalty (int, optional): penalty=1 (L1 norm), penalty=2 (L2 norm), and penalty=3 (Lipschitz norm). Defaults to 1.
        """
        self.eta = eta
        CQER.CQR.__init__(self, y, x, tau, z, cet, fun, rts)
        self.__model__.zeta = Var(
            self.__model__.I, self.__model__.J, doc='zeta')
        if penalty == 1:
            self.__model__.l1_norm1 = Constraint(self.__model__.I,
                                                 self.__model__.J,
                                                 rule=self.__l1_rule1(),
                                                 doc='L1 norm 1')
            self.__model__.l1_norm2 = Constraint(self.__model__.I,
                                                 self.__model__.J,
                                                 rule=self.__l1_rule2(),
                                                 doc='L1 norm 2')
            self.__model__.l1_norm3 = Constraint(rule=self.__l1_rule3(),
                                                 doc='L1 norm 3')

        elif penalty == 2:
            self.__model__.l2_norm = Constraint(rule=self.__l2_rule(),
                                                doc='L2 norm')
        elif penalty == 3:
            self.__model__.lipschitz_norm = Constraint(self.__model__.I,
                                                       rule=self.__lipschitz_rule(),
                                                       doc='Lipschitz norm')
        else:
            raise ValueError('Penalty must be 1, 2, or 3.')

    def __l1_rule1(self):
        """L1 norm #1: right-side equalities"""

        def l1_rule1(model, i, j):
            return model.beta[i, j] <= self.__model__.zeta[i, j]

        return l1_rule1

    def __l1_rule2(self):
        """L1 norm #2: left-side equalities"""

        def l1_rule2(model, i, j):
            return model.beta[i, j] >= -self.__model__.zeta[i, j]

        return l1_rule2

    def __l1_rule3(self):
        """L1 norm #3: sum of zeta"""

        def l1_rule3(model):
            return sum(self.__model__.zeta[ij] for ij in model.I * model.J) == self.eta

        return l1_rule3

    def __l2_rule(self):
        """L2 norm"""

        def l2_rule(model):
            return sum(model.beta[ij] ** 2 for ij in model.I * model.J) <= self.eta

        return l2_rule

    def __lipschitz_rule(self):
        """Lipschitz norm"""

        def lipschitz_rule(model, i):
            return sum(model.beta[i, j] ** 2 for j in model.J) <= self.eta ** 2

        return lipschitz_rule


class pCER(CQER.CER):
    """penalized Convex Expectile Regression (pCER)
    """

    def __init__(self, y, x, tau, eta, z=None, cet=CET_ADDI, fun=FUN_PROD, rts=RTS_VRS, penalty=1):
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
            penalty (int, optional): penalty=1 (L1 norm), penalty=2 (L2 norm), and penalty=3 (Lipschitz norm). Defaults to 1.
        """
        self.eta = eta
        CQER.CER.__init__(self, y, x, tau, z, cet, fun, rts)
        self.__model__.zeta = Var(
            self.__model__.I, self.__model__.J, doc='zeta')
        if penalty == 1:
            self.__model__.l1_norm1 = Constraint(self.__model__.I,
                                                 self.__model__.J,
                                                 rule=self.__l1_rule1(),
                                                 doc='L1 norm 1')
            self.__model__.l1_norm2 = Constraint(self.__model__.I,
                                                 self.__model__.J,
                                                 rule=self.__l1_rule2(),
                                                 doc='L1 norm 2')
            self.__model__.l1_norm3 = Constraint(rule=self.__l1_rule3(),
                                                 doc='L1 norm 3')

        elif penalty == 2:
            self.__model__.l2_norm = Constraint(rule=self.__l2_rule(),
                                                doc='L2 norm')
        elif penalty == 3:
            self.__model__.lipschitz_norm = Constraint(self.__model__.I,
                                                       rule=self.__lipschitz_rule(),
                                                       doc='Lipschitz norm')
        else:
            raise ValueError('Penalty must be 1, 2, or 3.')

    def __l1_rule1(self):
        """L1 norm #1: right-side equalities"""

        def l1_rule1(model, i, j):
            return model.beta[i, j] <= self.__model__.zeta[i, j]

        return l1_rule1

    def __l1_rule2(self):
        """L1 norm #2: left-side equalities"""

        def l1_rule2(model, i, j):
            return model.beta[i, j] >= -self.__model__.zeta[i, j]

        return l1_rule2

    def __l1_rule3(self):
        """L1 norm #3: sum of zeta"""

        def l1_rule3(model):
            return sum(self.__model__.zeta[ij] for ij in model.I * model.J) == self.eta

        return l1_rule3

    def __l2_rule(self):
        """L2 norm"""

        def l2_rule(model):
            return sum(model.beta[ij] ** 2 for ij in model.I * model.J) <= self.eta

        return l2_rule

    def __lipschitz_rule(self):
        """Lipschitz norm"""

        def lipschitz_rule(model, i):
            return sum(model.beta[i, j] ** 2 for j in model.J) <= self.eta ** 2

        return lipschitz_rule
