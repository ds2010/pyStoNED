# import dependencies
from . import CQER, ICNLS
from pyomo.environ import Constraint
from .constant import CET_ADDI, FUN_PROD, RTS_VRS


class ICQR(ICNLS.ICNLS, CQER.CQR):
    """Isotonic convex quantile regression (ICQR)
    """

    def __init__(self, y, x, tau, z=None, cet=CET_ADDI, fun=FUN_PROD, rts=RTS_VRS):
        """ICQR model

        Args:
             y (float): output variable. 
             x (float): input variables.
             tau (float): quantile.
             z (float, optional): Contextual variable(s). Defaults to None.
             cet (String, optional): CET_ADDI (additive composite error term) or CET_MULT (multiplicative composite error term). Defaults to CET_ADDI.
             fun (String, optional): FUN_PROD (production frontier) or FUN_COST (cost frontier). Defaults to FUN_PROD.
             rts (String, optional): RTS_VRS (variable returns to scale) or RTS_CRS (constant returns to scale). Defaults to RTS_VRS.
        """
        CQER.CQR.__init__(self, y, x, tau, z, cet, fun, rts)

        self._ICNLS__pmatrix = self._ICNLS__binaryMatrix()
        self.__model__.afriat_rule.deactivate()
        self.__model__.isotonic_afriat_rule = Constraint(self.__model__.I,
                                                         self.__model__.I,
                                                         rule=self._ICNLS__isotonic_afriat_rule(),
                                                         doc='isotonic afriat inequality')


class ICER(ICNLS.ICNLS, CQER.CER):
    """Isotonic convex expectile regression (ICER)"""

    def __init__(self, y, x, tau, z=None, cet=CET_ADDI, fun=FUN_PROD, rts=RTS_VRS):
        """ICER model

        Args:
             y (float): output variable. 
             x (float): input variables.
             tau (float): expectile.
             z (float, optional): Contextual variable(s). Defaults to None.
             cet (String, optional): CET_ADDI (additive composite error term) or CET_MULT (multiplicative composite error term). Defaults to CET_ADDI.
             fun (String, optional): FUN_PROD (production frontier) or FUN_COST (cost frontier). Defaults to FUN_PROD.
             rts (String, optional): RTS_VRS (variable returns to scale) or RTS_CRS (constant returns to scale). Defaults to RTS_VRS.
        """
        CQER.CER.__init__(self, y, x, tau, z, cet, fun, rts)

        self._ICNLS__pmatrix = self._ICNLS__binaryMatrix()
        self.__model__.afriat_rule.deactivate()
        self.__model__.isotonic_afriat_rule = Constraint(self.__model__.I,
                                                         self.__model__.I,
                                                         rule=self._ICNLS__isotonic_afriat_rule(),
                                                         doc='isotonic afriat inequality')
