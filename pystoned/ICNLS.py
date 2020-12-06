# Import CNLS as parent class
from . import CNLS
from pyomo.environ import Constraint
from pyomo.core.expr.numvalue import NumericValue
import numpy as np


class ICNLS(CNLS.CNLS):
    """Isotonic Convex Nonparametric Least Square (ICNLS)"""

    def __init__(self, y, x, z=None, cet='addi', fun='prod', rts='vrs'):
        """
        Initialize the ICNLS model

        * y: Output variable 
        * x: Input variables
        * z: Contextual variables              
        * cet = "addi" : Additive composite error term
              = "mult" : Multiplicative composite error term
        * fun = "prod" : Production frontier
              = "cost" : Cost frontier
        * rts = "vrs"  : Variable returns to scale
              = "crs"  : Constant returns to scale
        """
        super().__init__(y, x, z, cet, fun, rts)

        self.__pmatrix = self.__binaryMatrix()
        self.__model__.afriat_rule.deactivate()
        self.__model__.isotonic_afriat_rule = Constraint(self.__model__.I,
                                                         self.__model__.I,
                                                         rule=self.__isotonic_afriat_rule(),
                                                         doc='isotonic afriat inequality')

    def __isotonic_afriat_rule(self):
        """Return the proper afriat inequality constraint"""
        if self.fun == "prod":
            __operator = NumericValue.__le__
        elif self.fun == "cost":
            __operator = NumericValue.__ge__

        if self.cet == "addi":
            if self.rts == "vrs":

                def afriat_rule(model, i, h):
                    if i == h or self.__pmatrix[i][h] == 0:
                        return Constraint.Skip
                    return __operator(
                        self.__pmatrix[i][i]*(model.alpha[i] + sum(
                            model.beta[i, j] * self.x[i][j] for j in model.J)),
                        self.__pmatrix[i][h]*(model.alpha[h] + sum(
                            model.beta[h, j] * self.x[i][j] for j in model.J))
                    )
                return afriat_rule

            elif self.rts == "crs":
                # TODO(warning handling): replace with model requested not exist
                return False
        elif self.cet == "mult":
            if self.rts == "vrs":

                def afriat_rule(model, i, h):
                    if (i == h) or (self.__pmatrix[i][h] == 0):
                        return Constraint.Skip
                    return __operator(
                        self.__pmatrix[i][i]*(model.alpha[i] + sum(
                            model.beta[i, j] * self.x[i][j] for j in model.J)),
                        self.__pmatrix[i][h]*(model.alpha[h] + sum(
                            model.beta[h, j] * self.x[i][j] for j in model.J))
                    )
                return afriat_rule

            elif self.rts == "crs":

                def afriat_rule(model, i, h):
                    if (i == h) or (self.__pmatrix[i][h] == 0):
                        return Constraint.Skip
                    return __operator(
                        self.__pmatrix[i][i]*(sum(model.beta[i, j]
                                                  * self.x[i][j] for j in model.J)),
                        self.__pmatrix[i][h]*(sum(model.beta[h, j]
                                                  * self.x[i][j] for j in model.J))
                    )
                return afriat_rule

        # TODO(error handling): replace with undefined model attribute
        return False

    def __binaryMatrix(self):

        # transform data
        x = np.array(self.x)

        # number of DMUs
        n = len(x)
        m = len(x[0])

        # binary matrix P
        p = np.zeros((n, n))

        for i in range(n):
            pmap = (x[i, 0] <= x[:, 0])
            for j in range(1, m):
                pmap = pmap & (x[i, j] <= x[:, j])
            p[i, :] = np.where(pmap, 1, 0)

        return p.tolist()
