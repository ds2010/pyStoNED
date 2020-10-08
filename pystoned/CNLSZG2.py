# Import pyomo module
from pyomo.environ import ConcreteModel, Set, Var, Objective, minimize, Constraint, log
from pyomo.opt import SolverFactory, SolverManagerFactory
from pyomo.core.expr.numvalue import NumericValue
import numpy as np
import pandas as pd


class CNLSZG2:
    """CNLS+G in iterative loop"""

    def __init__(self, y, x, z, Cutactive, Active, cet='addi', fun='prod', rts='vrs'):
        """
            y : Output
            x : Input
            z : Contexutal variable
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
        self.z = z.tolist()
        self.cet = cet
        self.fun = fun
        self.rts = rts

        if type(self.x[0]) != list:
            self.x = []
            for x_value in x.tolist():
                self.x.append([x_value])

        if type(self.z[0]) != list:
            self.z = []
            for z_value in z.tolist():
                self.z.append([z_value])

        self.Cutactive = Cutactive
        self.Active = Active        

        # Initialize the CNLS model
        self.__model__ = ConcreteModel()

        # Initialize the sets
        self.__model__.I = Set(initialize=range(len(self.y)))
        self.__model__.J = Set(initialize=range(len(self.x[0])))
        self.__model__.K = Set(initialize=range(len(self.z[0])))

        # Initialize the variables
        self.__model__.alpha2 = Var(self.__model__.I, doc='alpha2')
        self.__model__.beta2 = Var(self.__model__.I,
                                   self.__model__.J,
                                   bounds=(0.0, None),
                                   doc='beta2')
        self.__model__.lamda2 = Var(self.__model__.K, doc='zvalue2')        
        self.__model__.epsilon2 = Var(self.__model__.I, doc='residual2')
        self.__model__.frontier2 = Var(self.__model__.I,
                                       bounds=(0.0, None),
                                       doc='estimated frontier2')

        # Setup the objective function and constraints
        self.__model__.objective2 = Objective(rule=self.__objective_rule2(),
                                              sense=minimize,
                                              doc='objective function2')
        self.__model__.regression_rule2 = Constraint(self.__model__.I,
                                                     rule=self.__regression_rule2(),
                                                     doc='regression equation2')
        if self.cet == "mult":
            self.__model__.log_rule2 = Constraint(self.__model__.I,
                                                  rule=self.__log_rule2(),
                                                  doc='log-transformed regression equation2')
        self.__model__.afriat_rule2 = Constraint(self.__model__.I,
                                                 rule=self.__afriat_rule2(),
                                                 doc='elementary Afriat approach2')
        self.__model__.sweet_rule2 = Constraint(self.__model__.I,
                                                self.__model__.I,
                                                rule=self.__sweet_rule2(),
                                                doc='sweet spot approach2')
        self.__model__.sweet2_rule2 = Constraint(self.__model__.I,
                                                 self.__model__.I,
                                                 rule=self.__sweet2_rule2(),
                                                 doc='sweet spot-2 approach2')

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

    def __objective_rule2(self):
        """Return the proper objective function"""

        def objective_rule2(model):
            return sum(model.epsilon2[i] ** 2 for i in model.I)

        return objective_rule2

    def __regression_rule2(self):
        """Return the proper regression constraint"""
        if self.cet == "addi":
            if self.rts == "vrs":

                def regression_rule2(model, i):
                    return self.y[i] == model.alpha2[i] + \
                           sum(model.beta2[i, j] * self.x[i][j] for j in model.J) + \
                           sum(model.lamda2[k] * self.z[i][k] for k in model.K) + model.epsilon2[i]

                return regression_rule2
            elif self.rts == "crs":
                # TODO(warning handling): replace with model requested not exist
                return False

        elif self.cet == "mult":

            def regression_rule2(model, i):
                return log(self.y[i]) == log(model.frontier2[i] + 1) + sum(model.lamda2[k] * self.z[i][k] for k in model.K) + \
                                          model.epsilon2[i]

            return regression_rule2

        # TODO(error handling): replace with undefined model attribute
        return False

    def __log_rule2(self):
        """Return the proper log constraint"""
        if self.cet == "mult":
            if self.rts == "vrs":

                def log_rule2(model, i):
                    return model.frontier2[i] == model.alpha2[i] + sum(
                        model.beta2[i, j] * self.x[i][j] for j in model.J) - 1

                return log_rule2
            elif self.rts == "crs":

                def log_rule2(model, i):
                    return model.frontier2[i] == sum(
                        model.beta2[i, j] * self.x[i][j] for j in model.J) - 1

                return log_rule2

        # TODO(error handling): replace with undefined model attribute
        return False

    def __afriat_rule2(self):
        """Return the proper elementary Afriat approach constraint"""
        if self.fun == "prod":
            __operator = NumericValue.__le__
        elif self.fun == "cost":
            __operator = NumericValue.__ge__

        if self.cet == "addi":
            if self.rts == "vrs":

                def afriat_rule2(model, i):
                    return __operator(
                        model.alpha2[i] + sum(model.beta2[i, j] * self.x[i][j]
                                              for j in model.J),
                        model.alpha2[self.__model__.I.nextw(i)] +
                        sum(model.beta2[self.__model__.I.nextw(i), j] * self.x[i][j]
                            for j in model.J))

                return afriat_rule2
            elif self.rts == "crs":
                # TODO(warning handling): replace with model requested not exist
                return False
        elif self.cet == "mult":
            if self.rts == "vrs":

                def afriat_rule2(model, i):
                    return __operator(
                        model.alpha2[i] + sum(model.beta2[i, j] * self.x[i][j]
                                              for j in model.J),
                        model.alpha2[self.__model__.I.nextw(i)] +
                        sum(model.beta2[self.__model__.I.nextw(i), j] * self.x[i][j]
                            for j in model.J))

                return afriat_rule2
            elif self.rts == "crs":

                def afriat_rule2(model, i):
                    return __operator(
                        sum(model.beta2[i, j] * self.x[i][j] for j in model.J),
                        sum(model.beta2[self.__model__.I.nextw(i), j] * self.x[i][j] for j in model.J))

                return afriat_rule2

        # TODO(error handling): replace with undefined model attribute
        return False

    def __sweet_rule2(self, ):
        """Return the proper sweet spot approach constraint"""
        if self.fun == "prod":
            __operator = NumericValue.__le__
        elif self.fun == "cost":
            __operator = NumericValue.__ge__

        if self.cet == "addi":
            if self.rts == "vrs":

                def sweet_rule2(model, i, h):
                    if self.Cutactive[i, h]:
                        if i == h:
                            return Constraint.Skip
                        return __operator(model.alpha2[i] + sum(model.beta2[i, j] * self.x[i][j]
                                                                for j in model.J),
                                          model.alpha2[h] + sum(model.beta2[h, j] * self.x[i][j]
                                                                for j in model.J))
                    return Constraint.Skip

                return sweet_rule2
            elif self.rts == "crs":
                # TODO(warning handling): replace with model requested not exist
                return False
        elif self.cet == "mult":
            if self.rts == "vrs":

                def sweet_rule2(model, i, h):
                    if self.Cutactive[i, h]:
                        if i == h:
                            return Constraint.Skip
                        return __operator(model.alpha2[i] + sum(model.beta2[i, j] * self.x[i][j]
                                                                for j in model.J),
                                          model.alpha2[h] + sum(model.beta2[h, j] * self.x[i][j]
                                                                for j in model.J))
                    return Constraint.Skip

                return sweet_rule2
            elif self.rts == "crs":

                def sweet_rule2(model, i, h):
                    if self.Cutactive[i, h]:
                        if i == h:
                            return Constraint.Skip
                        return __operator(sum(model.beta2[i, j] * self.x[i][j] for j in model.J),
                                          sum(model.beta2[h, j] * self.x[i][j] for j in model.J))
                    return Constraint.Skip

                return sweet_rule2

        # TODO(error handling): replace with undefined model attribute
        return False

    def __sweet2_rule2(self, ):
        """Return the proper sweet spot (step2) approach constraint"""
        if self.fun == "prod":
            __operator = NumericValue.__le__
        elif self.fun == "cost":
            __operator = NumericValue.__ge__

        if self.cet == "addi":
            if self.rts == "vrs":

                def sweet2_rule2(model, i, h):
                    if self.Active[i, h]:
                        if i == h:
                            return Constraint.Skip
                        return __operator(model.alpha2[i] + sum(model.beta2[i, j] * self.x[i][j]
                                                                for j in model.J),
                                          model.alpha2[h] + sum(model.beta2[h, j] * self.x[i][j]
                                                                for j in model.J))
                    return Constraint.Skip

                return sweet2_rule2
            elif self.rts == "crs":
                # TODO(warning handling): replace with model requested not exist
                return False
        elif self.cet == "mult":
            if self.rts == "vrs":

                def sweet2_rule2(model, i, h):
                    if self.Active[i, h]:
                        if i == h:
                            return Constraint.Skip
                        return __operator(model.alpha2[i] + sum(model.beta2[i, j] * self.x[i][j]
                                                                for j in model.J),
                                          model.alpha2[h] + sum(model.beta2[h, j] * self.x[i][j]
                                                                for j in model.J))
                    return Constraint.Skip

                return sweet2_rule2
            elif self.rts == "crs":

                def sweet2_rule2(model, i, h):
                    if self.Active[i, h]:
                        if i == h:
                            return Constraint.Skip
                        return __operator(sum(model.beta2[i, j] * self.x[i][j] for j in model.J),
                                          sum(model.beta2[h, j] * self.x[i][j] for j in model.J))
                    return Constraint.Skip

                return sweet2_rule2

        # TODO(error handling): replace with undefined model attribute
        return False

    def display_status(self):
        """Display the status of problem"""
        if self.optimization_status == 0:
            self.optimize()
        print(self.display_status)

    def display_alpha2(self):
        """Display alpha value"""
        if self.optimization_status == 0:
            self.optimize()
        self.__model__.alpha2.display()

    def display_beta2(self):
        """Display beta value"""
        if self.optimization_status == 0:
            self.optimize()
        self.__model__.beta2.display()

    def display_residual2(self):
        """Dispaly residual value"""
        if self.optimization_status == 0:
            self.optimize()
        self.__model__.epsilon2.display()

    def get_status(self):
        """Return status"""
        return self.optimization_status

    def get_alpha2(self):
        """Return alpha value by array"""
        if self.optimization_status == 0:
            self.optimize()
        alpha2 = list(self.__model__.alpha2[:].value)
        return np.asarray(alpha2)

    def get_beta2(self):
        """Return beta value by array"""
        if self.optimization_status == 0:
            self.optimize()
        beta2 = np.asarray([i + tuple([j]) for i, j in zip(list(self.__model__.beta2),
                                                           list(self.__model__.beta2[:, :].value))])
        beta2 = pd.DataFrame(beta2, columns=['Name', 'Key', 'Value'])
        beta2 = beta2.pivot(index='Name', columns='Key', values='Value')
        return beta2.to_numpy()

    def get_residual2(self):
        """Return residual value by array"""
        if self.optimization_status == 0:
            self.optimize()
        residual2 = list(self.__model__.epsilon2[:].value)
        return np.asarray(residual2)

    def get_lamda2(self):
        """Return residual value by array"""
        if self.optimization_status == 0:
            self.optimize()
        lamda2 = list(self.__model__.lamda2[:].value)
        return np.asarray(lamda2)
