# import dependencies
from . import pCQER, ICNLS
from pyomo.environ import Constraint
from .constant import CET_ADDI, FUN_PROD, RTS_VRS


class pICQR(ICNLS.ICNLS, pCQER.pCQR):
    """penalized Isotonic convex quantile regression (pICQR)
    """

    def __init__(self, y, x, tau, eta, z=None, cet=CET_ADDI, fun=FUN_PROD, rts=RTS_VRS):
        """pICQR model

        Args:
             y (float): output variable. 
             x (float): input variables.
             tau (float): quantile.
             eta (float): penalty parameter.
             z (float, optional): Contextual variable(s). Defaults to None.
             cet (String, optional): CET_ADDI (additive composite error term) or CET_MULT (multiplicative composite error term). Defaults to CET_ADDI.
             fun (String, optional): FUN_PROD (production frontier) or FUN_COST (cost frontier). Defaults to FUN_PROD.
             rts (String, optional): RTS_VRS (variable returns to scale) or RTS_CRS (constant returns to scale). Defaults to RTS_VRS.
        """
        pCQER.pCQR.__init__(self, y, x, tau, eta, z, cet, fun, rts)

        self._ICNLS__pmatrix = self._ICNLS__binaryMatrix()
        self.__model__.afriat_rule.deactivate()
        self.__model__.isotonic_afriat_rule = Constraint(self.__model__.I,
                                                         self.__model__.I,
                                                         rule=self._ICNLS__isotonic_afriat_rule(),
                                                         doc='isotonic afriat inequality')


class pICER(ICNLS.ICNLS, pCQER.pCER):
    """penalized Isotonic convex expectile regression (pICER)
    """

    def __init__(self, y, x, tau, eta, z=None, cet=CET_ADDI, fun=FUN_PROD, rts=RTS_VRS):
        """pICER model

        Args:
             y (float): output variable. 
             x (float): input variables.
             tau (float): expectile.
             eta (float): penalty parameter.
             z (float, optional): Contextual variable(s). Defaults to None.
             cet (String, optional): CET_ADDI (additive composite error term) or CET_MULT (multiplicative composite error term). Defaults to CET_ADDI.
             fun (String, optional): FUN_PROD (production frontier) or FUN_COST (cost frontier). Defaults to FUN_PROD.
             rts (String, optional): RTS_VRS (variable returns to scale) or RTS_CRS (constant returns to scale). Defaults to RTS_VRS.
        """
        pCQER.pCER.__init__(self, y, x, tau, eta, z, cet, fun, rts)

        self._ICNLS__pmatrix = self._ICNLS__binaryMatrix()
        self.__model__.afriat_rule.deactivate()
        self.__model__.isotonic_afriat_rule = Constraint(self.__model__.I,
                                                         self.__model__.I,
                                                         rule=self._ICNLS__isotonic_afriat_rule(),
                                                         doc='isotonic afriat inequality')
