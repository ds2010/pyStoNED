# Import pyomo module
from pyomo.environ import ConcreteModel, Set, Var, Objective, minimize, Constraint, log
from pyomo.opt import SolverFactory, SolverManagerFactory
from pyomo.core.expr.numvalue import NumericValue
import numpy as np
import pandas as pd


class CNLSG1:
    """initial Group-VC-added CNLS (CNLS+G) model"""

    def __init__(self, y, x, Cutactive, cet='addi', fun='prod', rts='vrs'):
        """
            y : Output variable
            x : Input variables
            cet  = "addi" : Additive composite error term
                 = "mult" : Multiplicative composite error term
            fun  = "prod" : Production frontier
                 = "cost" : Cost frontier
            rts  = "vrs"  : Variable returns to scale
                 = "crs"  : Constant returns to scale
        """

        # TODO(error/warning handling): Check the configuration of the model exist
        self.x = x.tolist()
        self.y = y.tolist()
        self.cet = cet
        self.fun = fun
        self.rts = rts

        if type(self.x[0]) != list:
            self.x = []
            for x_value in x.tolist():
                self.x.append([x_value])

        self.Cutactive = Cutactive

        # Initialize the CNLS model
        self.__model__ = ConcreteModel()

        # Initialize the sets
        self.__model__.I = Set(initialize=range(len(self.y)))
        self.__model__.J = Set(initialize=range(len(self.x[0])))

        # Initialize the variables
        self.__model__.alpha1 = Var(self.__model__.I, doc='alpha1')
        self.__model__.beta1 = Var(self.__model__.I,
                                   self.__model__.J,
                                   bounds=(0.0, None),
                                   doc='beta1')
        self.__model__.epsilon1 = Var(self.__model__.I, doc='residual1')
        self.__model__.frontier1 = Var(self.__model__.I,
                                       bounds=(0.0, None),
                                       doc='estimated frontier1')

        # Setup the objective function and constraints
        self.__model__.objective1 = Objective(rule=self.__objective_rule1(),
                                              sense=minimize,
                                              doc='objective function1')
        self.__model__.regression_rule1 = Constraint(self.__model__.I,
                                                     rule=self.__regression_rule1(),
                                                     doc='regression equation1')
        if self.cet == "mult":
            self.__model__.log_rule1 = Constraint(self.__model__.I,
                                                  rule=self.__log_rule1(),
                                                  doc='log-transformed regression equation1')
        self.__model__.afriat_rule1 = Constraint(self.__model__.I,
                                                 rule=self.__afriat_rule1(),
                                                 doc='elementary Afriat approach1')
        self.__model__.sweet_rule1 = Constraint(self.__model__.I,
                                                self.__model__.I,
                                                rule=self.__sweet_rule1(),
                                                doc='sweet spot approach1')

        # Optimize model
        self.optimization_status = 0
        self.problem_status = 0

    def optimize(self, remote=True):
        """Optimize the function by requested method"""
        # TODO(error/warning handling): Check problem status after optimization
        if remote == False:
            if self.cet == "addi":
                solver = SolverFactory("mosek")
                self.problem_status = solver.solve(self.__model__, tee=True)
                self.optimization_status = 1

            elif self.cet == "mult":
                # TODO(warning handling): Use log system instead of print()
                print(
                    "Estimating the multiplicative model will be available in near future."
                )
                return False
        else:
            if self.cet == "addi":
                opt = "mosek"

            elif self.cet == "mult":
                opt = "knitro"

            solver = SolverManagerFactory('neos')
            self.problem_status = solver.solve(self.__model__,
                                               tee=True,
                                               opt=opt)
            self.optimization_status = 1

    def __objective_rule1(self):
        """Return the proper objective function"""

        def objective_rule1(model):
            return sum(model.epsilon1[i] ** 2 for i in model.I)

        return objective_rule1

    def __regression_rule1(self):
        """Return the proper regression constraint"""
        if self.cet == "addi":
            if self.rts == "vrs":

                def regression_rule1(model, i):
                    return self.y[i] == model.alpha1[i] \
                           + sum(model.beta1[i, j] * self.x[i][j] for j in model.J) \
                           + model.epsilon1[i]

                return regression_rule1
            elif self.rts == "crs":
                ## TODO(warning handling): replace with model requested not exist
                return False

        elif self.cet == "mult":

            def regression_rule1(model, i):
                return log(
                    self.y[i]) == log(model.frontier1[i] + 1) + model.epsilon1[i]

            return regression_rule1

        # TODO(error handling): replace with undefined model attribute
        return False

    def __log_rule1(self):
        """Return the proper log constraint"""
        if self.cet == "mult":
            if self.rts == "vrs":

                def log_rule1(model, i):
                    return model.frontier1[i] == model.alpha1[i] + sum(
                        model.beta1[i, j] * self.x[i][j] for j in model.J) - 1

                return log_rule1
            elif self.rts == "crs":

                def log_rule1(model, i):
                    return model.frontier1[i] == sum(
                        model.beta1[i, j] * self.x[i][j] for j in model.J) - 1

                return log_rule1

        # TODO(error handling): replace with undefined model attribute
        return False

    def __afriat_rule1(self):
        """Return the proper elementary Afriat approach constraint"""
        if self.fun == "prod":
            __operator = NumericValue.__le__
        elif self.fun == "cost":
            __operator = NumericValue.__ge__

        if self.cet == "addi":
            if self.rts == "vrs":

                def afriat_rule1(model, i):
                    return __operator(
                        model.alpha1[i] + sum(model.beta1[i, j] * self.x[i][j]
                                              for j in model.J),
                        model.alpha1[self.__model__.I.nextw(i)] +
                        sum(model.beta1[self.__model__.I.nextw(i), j] * self.x[i][j]
                            for j in model.J))

                return afriat_rule1
            elif self.rts == "crs":
                # TODO(warning handling): replace with model requested not exist
                return False
        elif self.cet == "mult":
            if self.rts == "vrs":

                def afriat_rule1(model, i):
                    return __operator(
                        model.alpha1[i] + sum(model.beta1[i, j] * self.x[i][j]
                                              for j in model.J),
                        model.alpha1[self.__model__.I.nextw(i)] +
                        sum(model.beta1[self.__model__.I.nextw(i), j] * self.x[i][j]
                            for j in model.J))

                return afriat_rule1
            elif self.rts == "crs":

                def afriat_rule1(model, i):
                    return __operator(
                        sum(model.beta1[i, j] * self.x[i][j] for j in model.J),
                        sum(model.beta1[self.__model__.I.nextw(i), j] * self.x[i][j] for j in model.J))

                return afriat_rule1

        # TODO(error handling): replace with undefined model attribute
        return False

    def __sweet_rule1(self, ):
        """Return the proper sweet spot approach constraint"""
        if self.fun == "prod":
            __operator = NumericValue.__le__
        elif self.fun == "cost":
            __operator = NumericValue.__ge__

        if self.cet == "addi":
            if self.rts == "vrs":

                def sweet_rule1(model, i, h):
                    if self.Cutactive[i, h]:
                        if i == h:
                            return Constraint.Skip
                        return __operator(model.alpha1[i] + sum(model.beta1[i, j] * self.x[i][j]
                                                                for j in model.J),
                                          model.alpha1[h] + sum(model.beta1[h, j] * self.x[i][j]
                                                                for j in model.J))
                    return Constraint.Skip

                return sweet_rule1
            elif self.rts == "crs":
                # TODO(warning handling): replace with model requested not exist
                return False
        elif self.cet == "mult":
            if self.rts == "vrs":

                def sweet_rule1(model, i, h):
                    if self.Cutactive[i, h]:
                        if i == h:
                            return Constraint.Skip
                        return __operator(model.alpha1[i] + sum(model.beta1[i, j] * self.x[i][j]
                                                                for j in model.J),
                                          model.alpha1[h] + sum(model.beta1[h, j] * self.x[i][j]
                                                                for j in model.J))
                    return Constraint.Skip

                return sweet_rule1
            elif self.rts == "crs":

                def sweet_rule1(model, i, h):
                    if self.Cutactive[i, h]:
                        if i == h:
                            return Constraint.Skip
                        return __operator(sum(model.beta1[i, j] * self.x[i][j] for j in model.J),
                                          sum(model.beta1[h, j] * self.x[i][j] for j in model.J))
                    return Constraint.Skip

                return sweet_rule1

        # TODO(error handling): replace with undefined model attribute
        return False

    def display_status(self):
        """Display the status of problem"""
        if self.optimization_status == 0:
            self.optimize()
        print(self.display_status)

    def display_alpha1(self):
        """Display alpha value"""
        if self.optimization_status == 0:
            self.optimize()
        self.__model__.alpha1.display()

    def display_beta1(self):
        """Display beta value"""
        if self.optimization_status == 0:
            self.optimize()
        self.__model__.beta1.display()

    def display_residual1(self):
        """Dispaly residual value"""
        if self.optimization_status == 0:
            self.optimize()
        self.__model__.epsilon1.display()

    def get_status(self):
        """Return status"""
        return self.optimization_status

    def get_alpha1(self):
        """Return alpha value by array"""
        if self.optimization_status == 0:
            self.optimize()
        alpha1 = list(self.__model__.alpha1[:].value)
        return np.asarray(alpha1)

    def get_beta1(self):
        """Return beta value by array"""
        if self.optimization_status == 0:
            self.optimize()
        beta1 = np.asarray([i + tuple([j]) for i, j in zip(list(self.__model__.beta1),
                                                           list(self.__model__.beta1[:, :].value))])
        beta1 = pd.DataFrame(beta1, columns=['Name', 'Key', 'Value'])
        beta1 = beta1.pivot(index='Name', columns='Key', values='Value')
        return beta1.to_numpy()

    def get_residual1(self):
        """Return residual value by array"""
        if self.optimization_status == 0:
            self.optimize()
        residual1 = list(self.__model__.epsilon1[:].value)
        return np.asarray(residual1)
