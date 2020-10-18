# Import CNLS as parent class
from . import CQER, ICNLS
from pyomo.environ import Constraint
from pyomo.core.expr.numvalue import NumericValue
import numpy as np


class ICQR(ICNLS.ICNLS, CQER.CQR):
    """Isotonic convex quantile regression (ICQR)"""

    def __init__(self, y, x, tau, cet='addi', fun='prod', rts='vrs'):
        """
            y : Output variable
            x : Input variables
            tau : quantile
            cet  = "addi" : Additive composite error term
                 = "mult" : Multiplicative composite error term
            fun  = "prod" : Production frontier
                 = "cost" : Cost frontier
            rts  = "vrs"  : Variable returns to scale
                 = "crs"  : Constant returns to scale
        """
        CQER.CQR.__init__(self, y, x, tau, cet, fun, rts)

        self._ICNLS__pmatrix = self._ICNLS__binaryMatrix()
        self.__model__.afriat_rule.deactivate()
        self.__model__.isotonic_afriat_rule = Constraint(self.__model__.I,
                                                         self.__model__.I,
                                                         rule=self._ICNLS__isotonic_afriat_rule(),
                                                         doc='isotonic afriat inequality')


class ICER(ICNLS.ICNLS, CQER.CER):
    """Isotonic convex expectile regression (ICER)"""

    def __init__(self, y, x, tau, cet='addi', fun='prod', rts='vrs'):
        """
            y : Output variable
            x : Input variables
            tau : expectile
            cet  = "addi" : Additive composite error term
                 = "mult" : Multiplicative composite error term
            fun  = "prod" : Production frontier
                 = "cost" : Cost frontier
            rts  = "vrs"  : Variable returns to scale
                 = "crs"  : Constant returns to scale
        """
        CQER.CER.__init__(self, y, x, tau, cet, fun, rts)

        self._ICNLS__pmatrix = self._ICNLS__binaryMatrix()
        self.__model__.afriat_rule.deactivate()
        self.__model__.isotonic_afriat_rule = Constraint(self.__model__.I,
                                                         self.__model__.I,
                                                         rule=self._ICNLS__isotonic_afriat_rule(),
                                                         doc='isotonic afriat inequality')
