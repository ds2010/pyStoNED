# import dependencies
from pyomo.environ import ConcreteModel, Set, Var, Objective, minimize, Constraint, log
from pyomo.core.expr.numvalue import NumericValue
import numpy as np
import pandas as pd

from . import CNLS
from .constant import CET_MULT, RDF_DI, RDF_DO, OPT_DEFAULT, RTS_CRS, RTS_VRS, OPT_LOCAL
from .utils import tools


class CNLSRDF(CNLS.CNLS):
    """Convex Nonparametric Least Square with radial distance function
    """

    def __init__(self, y, x, z=None, rdf=RDF_DI, rts=RTS_VRS):
        """CNLSRDF model
        Args:
            y (float): output variable. 
            x (float): input variables.
            z (float, optional): control variables. Defaults to None.
            cet (String, optional): CET_ADDI (additive composite error term) or CET_MULT (multiplicative composite error term). Defaults to CET_ADDI.
            rdf (String, optional): RDF_DI (input distance function) or RDF_DO (output distance function). Defaults to RDF_DI.
            rts (String, optional): RTS_VRS (variable returns to scale) or RTS_CRS (constant returns to scale). Defaults to RTS_VRS.
        """
        # TODO(error/warning handling): Check the configuration of the model exist
        self.y, self.x, self.z = tools.assert_valid_mupltiple_x_y_data(y, x, z)
        self.rdf, self.rts = rdf, rts

        # rescale the input/output variables
        if self.rdf == RDF_DI:
            self.X = np.array(self.x)
            self.X = np.cumprod(self.X**(1/self.X.shape[1]), axis=1)[:, -1]
            self.X = tools.to_1d_list(tools.trans_list(self.X))

        elif self.rdf == RDF_DO:
            self.Y = np.array(self.y)
            self.Y = np.cumprod(self.Y**(1/self.Y.shape[1]), axis=1)[:, -1]
            self.Y = tools.to_1d_list(tools.trans_list(self.Y))

        # Initialize the CNLS model
        self.__model__ = ConcreteModel()

        if type(self.z) != type(None):
            # Initialize the set of z
            self.__model__.K = Set(initialize=range(len(self.z[0])))

            # Initialize the variables for z variable
            self.__model__.lamda = Var(self.__model__.K, doc='z coefficient')

        # Initialize the sets
        self.__model__.I = Set(initialize=range(len(self.y)))
        self.__model__.J = Set(initialize=range(len(self.x[0])))
        self.__model__.Q = Set(initialize=range(len(self.y[0])))
        if self.rdf == RDF_DI:
            self.__model__.JJ = Set(initialize=range(1, len(self.x[0])))
        elif self.rdf == RDF_DO:
            self.__model__.QQ = Set(initialize=range(1, len(self.y[0])))

        # Initialize the variables
        self.__model__.alpha = Var(self.__model__.I, doc='alpha')
        self.__model__.beta = Var(self.__model__.I,
                                  self.__model__.J,
                                  bounds=(0.0, None),
                                  doc='beta')
        self.__model__.epsilon = Var(self.__model__.I, doc='residual')
        self.__model__.chi = Var(self.__model__.I,
                                 bounds=(0.0, None),
                                 doc='estimated radial distance function')
        self.__model__.gamma = Var(self.__model__.I,
                                   self.__model__.Q,
                                   bounds=(0.0, None),
                                   doc='gamma')

        if self.rdf == RDF_DI:
            self.__model__.kappa = Var(self.__model__.JJ, doc='free parameter')
        elif self.rdf == RDF_DO:
            self.__model__.kappa = Var(self.__model__.QQ, doc='free parameter')

        # Setup the objective function and constraints
        self.__model__.objective = Objective(rule=self._CNLS__objective_rule(),
                                             sense=minimize,
                                             doc='objective function')
        self.__model__.regression_rule = Constraint(self.__model__.I,
                                                    rule=self.__regression_rule(),
                                                    doc='regression equation')
        self.__model__.log_rule = Constraint(self.__model__.I,
                                             rule=self.__log_rule(),
                                             doc='log-transformed regression equation')
        self.__model__.afriat_rule = Constraint(self.__model__.I,
                                                self.__model__.I,
                                                rule=self.__afriat_rule(),
                                                doc='afriat inequality')

        # Optimize model
        self.optimization_status, self.problem_status = 0, 0

    def optimize(self, email=OPT_LOCAL, solver=OPT_DEFAULT):
        """Optimize the function by requested method
        Args:
            email (string): The email address for remote optimization. It will optimize locally if OPT_LOCAL is given.
            solver (string): The solver chosen for optimization. It will optimize with default solver if OPT_DEFAULT is given.
        """
        # TODO(error/warning handling): Check problem status after optimization
        self.problem_status, self.optimization_status = tools.optimize_model(
            self.__model__, email, CET_MULT, solver)

    def __regression_rule(self):
        """Return the proper regression constraint"""
        if self.rdf == RDF_DI:

            if type(self.z) != type(None):
                def regression_rule(model, i):
                    return log(self.x[i][0]) == -log(model.chi[i] + 1) + sum(model.kappa[j] *
                                                                             (log(
                                                                                 self.x[i][0]) - log(self.x[i][j]))
                                                                             for j in model.JJ) + sum(model.lamda[k] * self.z[i][k]
                                                                                                      for k in model.K) + model.epsilon[i]

                return regression_rule

            def regression_rule(model, i):
                return log(self.x[i][0]) == -log(model.chi[i] + 1) + sum(model.kappa[j] *
                                                                         (log(
                                                                             self.x[i][0]) - log(self.x[i][j]))
                                                                         for j in model.JJ) + model.epsilon[i]

            return regression_rule
        elif self.rdf == RDF_DO:

            if type(self.z) != type(None):
                def regression_rule(model, i):
                    return log(self.y[i][0]) == -log(model.chi[i] + 1) + sum(model.kappa[q] *
                                                                             (log(
                                                                                 self.y[i][0]) - log(self.y[i][q]))
                                                                             for q in model.QQ) + sum(model.lamda[k] * self.z[i][k]
                                                                                                      for k in model.K) + model.epsilon[i]

                return regression_rule

            def regression_rule(model, i):
                return log(self.y[i][0]) == -log(model.chi[i] + 1) + sum(model.kappa[q] *
                                                                         (log(
                                                                             self.y[i][0]) - log(self.y[i][q]))
                                                                         for q in model.QQ) + model.epsilon[i]

            return regression_rule

        raise ValueError("Undefined model parameters.")

    def __log_rule(self):
        """Return the proper log constraint"""
        if self.rdf == RDF_DI:
            if self.rts == RTS_VRS:

                def log_rule(model, i):
                    return model.chi[i] == model.alpha[i] + sum(
                        model.beta[i, j] * (self.x[i][j]/self.X[i]) for j in model.J) \
                        - sum(model.gamma[i, q] * self.y[i][q]
                              for q in model.Q) - 1

                return log_rule
            elif self.rts == RTS_CRS:

                def log_rule(model, i):
                    return model.chi[i] == sum(
                        model.beta[i, j] * (self.x[i][j]/self.X[i]) for j in model.J) \
                        - sum(model.gamma[i, q] * self.y[i][q]
                              for q in model.Q) - 1

                return log_rule

        elif self.rdf == RDF_DO:
            if self.rts == RTS_VRS:

                def log_rule(model, i):
                    return model.chi[i] == model.alpha[i] + sum(
                        model.gamma[i, q] * (self.y[i][q]/self.Y[i]) for q in model.Q) \
                        - sum(model.beta[i, j] * self.x[i][j]
                              for j in model.J) - 1

                return log_rule
            elif self.rts == RTS_CRS:

                def log_rule(model, i):
                    return model.chi[i] == sum(
                        model.gamma[i, q] * (self.y[i][q]/self.Y[i]) for q in model.Q) \
                        - sum(model.beta[i, j] * self.x[i][j]
                              for j in model.J) - 1

                return log_rule

        raise ValueError("Undefined model parameters.")

    def __afriat_rule(self):
        """Return the proper afriat inequality constraint"""
        if self.rdf == RDF_DI:
            __operator = NumericValue.__le__
        elif self.rdf == RDF_DO:
            __operator = NumericValue.__ge__

        if self.rdf == RDF_DI:
            if self.rts == RTS_VRS:

                def afriat_rule(model, i, h):
                    if i == h:
                        return Constraint.Skip
                    return __operator(
                        model.alpha[i] + sum(model.beta[i, j] * (self.x[i][j]/self.X[i])
                                             for j in model.J) - sum(model.gamma[i, q] * self.y[i][q]
                                                                     for q in model.Q),
                        model.alpha[h] + sum(model.beta[h, j] * (self.x[i][j]/self.X[i])
                                             for j in model.J) - sum(model.gamma[h, q] * self.y[i][q]
                                                                     for q in model.Q))

                return afriat_rule
            elif self.rts == RTS_CRS:

                def afriat_rule(model, i, h):
                    if i == h:
                        return Constraint.Skip
                    return __operator(
                        sum(model.beta[i, j] * (self.x[i][j]/self.X[i])
                            for j in model.J) - sum(model.gamma[i, q] * self.y[i][q]
                                                    for q in model.Q),
                        sum(model.beta[h, j] * (self.x[i][j]/self.X[i])
                            for j in model.J) - sum(model.gamma[h, q] * self.y[i][q]
                                                    for q in model.Q))

                return afriat_rule

        elif self.rdf == RDF_DO:
            if self.rts == RTS_VRS:

                def afriat_rule(model, i, h):
                    if i == h:
                        return Constraint.Skip
                    return __operator(
                        model.alpha[i] + sum(model.gamma[i, q] * (self.y[i][q]/self.Y[i])
                                             for q in model.Q) - sum(model.beta[i, j] * self.x[i][j]
                                                                     for j in model.J),
                        model.alpha[h] + sum(model.gamma[h, q] * (self.y[i][q]/self.Y[i])
                                             for q in model.Q) - sum(model.beta[h, j] * self.x[i][j]
                                                                     for j in model.J))

                return afriat_rule
            elif self.rts == RTS_CRS:

                def afriat_rule(model, i, h):
                    if i == h:
                        return Constraint.Skip
                    return __operator(
                        sum(model.gamma[i, q] * (self.y[i][q]/self.Y[i])
                            for q in model.Q) - sum(model.beta[i, j] * self.x[i][j]
                                                    for j in model.J),
                        sum(model.gamma[h, q] * (self.y[i][q]/self.Y[i])
                            for q in model.Q) - sum(model.beta[h, j] * self.x[i][j]
                                                    for j in model.J))

                return afriat_rule

        raise ValueError("Undefined model parameters.")

    def display_gamma(self):
        """Display gamma value"""
        tools.assert_optimized(self.optimization_status)
        self.__model__.gamma.display()

    def get_gamma(self):
        """Return gamma value by array"""
        tools.assert_optimized(self.optimization_status)
        gamma = np.asarray([i + tuple([j]) for i, j in zip(list(self.__model__.gamma),
                                                           list(self.__model__.gamma[:, :].value))])
        gamma = pd.DataFrame(gamma, columns=['Name', 'Key', 'Value'])
        gamma = gamma.pivot(index='Name', columns='Key', values='Value')
        return gamma.to_numpy()

    def display_kappa(self):
        """Display kappa value"""
        tools.assert_optimized(self.optimization_status)
        self.__model__.kappa.display()

    def get_kappa(self):
        """Return kappa value by array"""
        tools.assert_optimized(self.optimization_status)
        kappa = list(self.__model__.kappa[:].value)
        return np.asarray(kappa)

    def get_chi(self):
        """Return estimated radial distance function by array"""
        tools.assert_optimized(self.optimization_status)
        chi = np.asarray(list(self.__model__.chi[:].value)) + 1
        return np.asarray(chi)
