# Import pyomo module
from pyomo.environ import ConcreteModel, Set, Var, Objective, minimize, Constraint, log
from pyomo.opt import SolverFactory, SolverManagerFactory
from pyomo.core.expr.numvalue import NumericValue


class CNLS:
    def __init__(self, y, x, cet='addi', fun='prod', rts='vrs'):
        """
        Convex Nonparametric Least Square (CNLS)
            y : Output variable
            x : Input variables
            cet  = "addi" : Additive composite error term
                 = "mult" : Multiplicative composite error term
            fun  = "prod" : production frontier
                 = "cost" : cost frontier
            rts  = "vrs"  : variable returns to scale
                 = "crs"  : constant returns to scale
        """

        ## TODO(error/warning handling): Check the configuration of the model exist
        self.x = x.tolist()
        self.y = y.tolist()
        self.cet = cet
        self.fun = fun
        self.rts = rts

        if type(self.x[0]) != list:
            self.x = [x.tolist()]

        # Initialize the CNLS model
        self.__model__ = ConcreteModel()

        # Initialize the sets
        self.__model__.I = Set(initialize=range(len(self.y)))
        self.__model__.J = Set(initialize=range(len(self.x[0])))

        # Initialize the variables
        self.__model__.alpha = Var(self.__model__.I, doc='alpha')
        self.__model__.beta = Var(self.__model__.I,
                                  self.__model__.J,
                                  bounds=(0.0, None),
                                  doc='beta')
        self.__model__.epsilon = Var(self.__model__.I, doc='residual')
        self.__model__.frontier = Var(self.__model__.I,
                                      bounds=(0.0, None),
                                      doc='estimated frontier')

        # Setup the objective function and constraints
        self.__model__.objective = Objective(rule=self.__objective_rule(),
                                             sense=minimize,
                                             doc='Objective function')
        self.__model__.regression_rule = Constraint(self.__model__.I,
                                                    rule=self.__regression_rule(),
                                                    doc='Regression equation')
        if self.cet == "mult":
            self.__model__.log_rule = Constraint(self.__model__.I,
                                                 rule=self.__log_rule(),
                                                 doc='Log-transformed regression equation')
        self.__model__.afriat_rule = Constraint(self.__model__.I,
                                                self.__model__.I,
                                                rule=self.__afriat_rule(),
                                                doc='Afriat inequality')

        # Optimize model
        self.optimization_status = 0
        self.problem_status = 0

    def optimize(self, remote=True):
        """Optimize the function by requested method"""
        ## TODO(error/warning handling): Check problem status after optimization
        if remote == False:
            if self.cet == "addi":
                solver = SolverFactory("mosek")
                self.problem_status = solver.solve(self.__model__, tee=True)
                self.optimization_status = 1

            elif self.cet == "mult":
                ## TODO(warning handling): Use log system instead of print()
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

    def __objective_rule(self):
        """Return the proper objective function for config"""
        def objective_rule(model):
            return sum(model.epsilon[i]**2 for i in model.I)

        return objective_rule

    def __regression_rule(self):
        """Return the proper regression constraint for config"""
        if self.cet == "addi":
            if self.rts == "vrs":

                def regression_rule(model, i):
                    return self.y[i] == model.alpha[i] \
                        + sum(model.beta[i, j] * self.x[i][j] for j in model.J)\
                        + model.epsilon[i]

                return regression_rule
            elif self.rts == "crs":
                ## TODO(warning handling): replace with model requested not exist
                return False

        elif self.cet == "mult":

            def regression_rule(model, i):
                return log(
                    self.y[i]) == log(model.frontier[i] + 1) + model.epsilon[i]

            return regression_rule

        ## TODO(error handling): replace with undefined model attribute
        return False

    def __log_rule(self):
        """Return the proper log constraint for config"""
        if self.cet == "mult":
            if self.rts == "vrs":

                def log_rule(model, i):
                    return model.frontier[i] == model.alpha[i] + sum(
                        model.beta[i, j] * self.x[i][j] for j in model.J) - 1

                return log_rule
            elif self.rts == "crs":

                def log_rule(model, i):
                    return model.frontier[i] == sum(
                        model.beta[i, j] * self.x[i][j] for j in model.J) - 1

                return log_rule

        ## TODO(error handling): replace with undefined model attribute
        return False

    def __afriat_rule(self, ):
        """Return the proper afriat inequality constraint for config"""
        if self.fun == "prod":
            __operator = NumericValue.__le__
        elif self.fun == "cost":
            __operator = NumericValue.__ge__

        if self.cet == "addi":
            if self.rts == "vrs":

                def afriat_rule(model, i, h):
                    if i == h:
                        return Constraint.Skip
                    return __operator(
                        model.alpha[i] + sum(model.beta[i, j] * self.x[i][j]
                                             for j in model.J),
                        model.alpha[h] + sum(model.beta[h, j] * self.x[i][j]
                                             for j in model.J))

                return afriat_rule
            elif self.rts == "crs":
                ## TODO(warning handling): replace with model requested not exist
                return False
        elif self.cet == "mult":
            if self.rts == "vrs":

                def afriat_rule(model, i, h):
                    if i == h:
                        return Constraint.Skip
                    return __operator(
                        model.alpha[i] + sum(model.beta[i, j] * self.x[i][j]
                                             for j in model.J),
                        model.alpha[h] + sum(model.beta[h, j] * self.x[i][j]
                                             for j in model.J))

                return afriat_rule
            elif self.rts == "crs":

                def afriat_rule(model, i, h):
                    if i == h:
                        return Constraint.Skip
                    return __operator(
                        sum(model.beta[i, j] * self.x[i][j] for j in model.J),
                        sum(model.beta[h, j] * self.x[i][j] for j in model.J))

                return afriat_rule

        ## TODO(error handling): replace with undefined model attribute
        return False

    def display_status(self):
        """Display the status of problem"""
        if self.optimization_status == 0:
            self.optimize()
        print(self.display_status)

    def display_alpha(self):
        """Display alpha value"""
        if self.optimization_status == 0:
            self.optimize()
        self.__model__.alpha.display()

    def display_beta(self):
        """Display beta value"""
        if self.optimization_status == 0:
            self.optimize()
        self.__model__.beta.display()

    def display_residual(self):
        """Dispaly residual value"""
        if self.optimization_status == 0:
            self.optimize()
        self.__model__.epsilon.display()

    def get_status(self):
        """Return status"""
        return self.optimization_status

    def get_alpha(self):
        """Return alpha value by list"""
        if self.optimization_status == 0:
            self.optimize()
        return list(self.__model__.alpah[:].value)

    def get_beta(self):
        """Return beta value by list"""
        if self.optimization_status == 0:
            self.optimize()
        return list(self.__model__.beta[:].value)

    def get_residual(self):
        """Return residual value by list"""
        if self.optimization_status == 0:
            self.optimize()
        return list(self.__model__.epsilon[:].value)