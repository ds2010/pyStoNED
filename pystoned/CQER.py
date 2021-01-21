# Import pyomo module
from pyomo.environ import ConcreteModel, Set, Var, Objective, minimize, Constraint, log
from pyomo.opt import SolverFactory, SolverManagerFactory
from pyomo.core.expr.numvalue import NumericValue
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class CQR:
    """Convex quantile regression (CQR)"""

    def __init__(self, y, x, tau, z=None, cet='addi', fun='prod', rts='vrs'):
        """
          * y : Output variable
          * x : Input variables
          * z: Contextual variables  
          * tau : quantile
          * cet  = "addi" : Additive composite error term
                 = "mult" : Multiplicative composite error term
          * fun  = "prod" : Production frontier
                 = "cost" : Cost frontier
          * rts  = "vrs"  : Variable returns to scale
                 = "crs"  : Constant returns to scale
        """

        # TODO(error/warning handling): Check the configuration of the model exist
        self.x = x.tolist()
        self.y = y.tolist()
        self.z = z
        self.tau = tau
        self.cet = cet
        self.fun = fun
        self.rts = rts

        if type(self.x[0]) != list:
            self.x = []
            for x_value in x.tolist():
                self.x.append([x_value])

        if type(self.y[0]) == list:
            self.y = self.__to_1d_list(self.y)

        # Initialize the CQR model
        self.__model__ = ConcreteModel()

        if type(self.z) != type(None):
            self.z = z.tolist()
            if type(self.z[0]) != list:
                self.z = []
                for z_value in z.tolist():
                    self.z.append([z_value])

            # Initialize the set of z
            self.__model__.K = Set(initialize=range(len(self.z[0])))

            # Initialize the variables for z variable
            self.__model__.lamda = Var(self.__model__.K, doc='z coefficient')

        # Initialize the sets
        self.__model__.I = Set(initialize=range(len(self.y)))
        self.__model__.J = Set(initialize=range(len(self.x[0])))

        # Initialize the variables
        self.__model__.alpha = Var(self.__model__.I, doc='alpha')
        self.__model__.beta = Var(self.__model__.I,
                                self.__model__.J,
                                bounds=(0.0, None),
                                doc='beta')
        self.__model__.epsilon = Var(self.__model__.I, doc='error term')
        self.__model__.epsilon_plus = Var(
            self.__model__.I, bounds=(0.0, None), doc='positive error term')
        self.__model__.epsilon_minus = Var(
            self.__model__.I, bounds=(0.0, None), doc='negative error term')
        self.__model__.frontier = Var(self.__model__.I,
                                    bounds=(0.0, None),
                                    doc='estimated frontier')

        # Setup the objective function and constraints
        self.__model__.objective = Objective(rule=self.__objective_rule(),
                                            sense=minimize,
                                            doc='objective function')

        self.__model__.error_decomposition = Constraint(self.__model__.I,
                                                        rule=self.__error_decomposition(),
                                                        doc='decompose error term')

        self.__model__.regression_rule = Constraint(self.__model__.I,
                                                    rule=self.__regression_rule(),
                                                    doc='regression equation')
        if self.cet == "mult":
            self.__model__.log_rule = Constraint(self.__model__.I,
                                                    rule=self.__log_rule(),
                                                    doc='log-transformed regression equation')

        self.__model__.afriat_rule = Constraint(self.__model__.I,
                                                self.__model__.I,
                                                rule=self.__afriat_rule(),
                                                doc='afriat inequality')

        # Optimize model
        self.optimization_status = 0
        self.problem_status = 0

    def optimize(self, remote=True):
        """Optimize the function by requested method"""
        # TODO(error/warning handling): Check problem status after optimization
        if remote == False:
            if self.cet == "addi":
                solver = SolverFactory("mosek")
                print("Estimating the additive model locally with mosek solver")
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
                print("Estimating the additive model remotely with mosek solver")
            elif self.cet == "mult":
                opt = "knitro"
                print("Estimating the multiplicative model remotely with knitro solver")
            solver = SolverManagerFactory('neos')
            self.problem_status = solver.solve(self.__model__,
                                                tee=True,
                                                opt=opt)
            self.optimization_status = 1

    def __to_1d_list(self, l):
        rl = []
        for i in range(len(l)):
            rl.append(l[i][0])
        return rl

    def __objective_rule(self):
        """Return the proper objective function"""

        def objective_rule(model):
            return self.tau * sum(model.epsilon_plus[i] for i in model.I) \
                    + (1 - self.tau) * sum(model.epsilon_minus[i] for i in model.I)

        return objective_rule

    def __error_decomposition(self):
        """Return the constraint decomposing error to positive and negative terms"""

        def error_decompose_rule(model, i):
            return model.epsilon[i] == model.epsilon_plus[i] - model.epsilon_minus[i]

        return error_decompose_rule

    def __regression_rule(self):
        """Return the proper regression constraint"""
        if self.cet == "addi":
            if self.rts == "vrs":
                if type(self.z) != type(None):
                    def regression_rule(model, i):
                        return self.y[i] == model.alpha[i] \
                            + sum(model.beta[i, j] * self.x[i][j] for j in model.J) \
                            + sum(model.lamda[k] * self.z[i][k]
                                    for k in model.K) + model.epsilon[i]
                    return regression_rule

                def regression_rule(model, i):
                    return self.y[i] == model.alpha[i] \
                        + sum(model.beta[i, j] * self.x[i][j] for j in model.J) \
                        + model.epsilon[i]

                return regression_rule
            elif self.rts == "crs":
                # TODO(warning handling): replace with model requested not exist
                return False

        elif self.cet == "mult":
            if type(self.z) != type(None):
                def regression_rule(model, i):
                    return log(self.y[i]) == log(model.frontier[i] + 1) \
                                + sum(model.lamda[k] * self.z[i][k]
                                        for k in model.K) + model.epsilon[i]
                return regression_rule

            def regression_rule(model, i):
                return log(self.y[i]) == log(model.frontier[i] + 1) + model.epsilon[i]
            return regression_rule

        # TODO(error handling): replace with undefined model attribute
        return False

    def __log_rule(self):
        """Return the proper log constraint"""
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

        # TODO(error handling): replace with undefined model attribute
        return False

    def __afriat_rule(self):
        """Return the proper afriat inequality constraint"""
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
                # TODO(warning handling): replace with model requested not exist
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

        # TODO(error handling): replace with undefined model attribute
        return False

    def display_status(self):
        """Display the status of problem"""
        if self.optimization_status == 0:
            print("Model isn't optimized. Use optimize() method to estimate the model.")
            return False
        print(self.display_status)

    def display_alpha(self):
        """Display alpha value"""
        if self.optimization_status == 0:
            print("Model isn't optimized. Use optimize() method to estimate the model.")
            return False
        self.__model__.alpha.display()

    def display_beta(self):
        """Display beta value"""
        if self.optimization_status == 0:
            print("Model isn't optimized. Use optimize() method to estimate the model.")
            return False
        self.__model__.beta.display()

    def display_residual(self):
        """Dispaly residual value"""
        if self.optimization_status == 0:
            print("Model isn't optimized. Use optimize() method to estimate the model.")
            return False
        self.__model__.epsilon.display()

    def display_positive_residual(self):
        """Dispaly positive residual value"""
        if self.optimization_status == 0:
            print("Model isn't optimized. Use optimize() method to estimate the model.")
            return False
        self.__model__.epsilon_plus.display()

    def display_negative_residual(self):
        """Dispaly negative residual value"""
        if self.optimization_status == 0:
            print("Model isn't optimized. Use optimize() method to estimate the model.")
            return False
        self.__model__.epsilon_minus.display()

    def get_status(self):
        """Return status"""
        return self.optimization_status

    def get_alpha(self):
        """Return alpha value by array"""
        if self.optimization_status == 0:
            print("Model isn't optimized. Use optimize() method to estimate the model.")
            return False
        alpha = list(self.__model__.alpha[:].value)
        return np.asarray(alpha)

    def get_beta(self):
        """Return beta value by array"""
        if self.optimization_status == 0:
            print("Model isn't optimized. Use optimize() method to estimate the model.")
            return False
        beta = np.asarray([i + tuple([j]) for i, j in zip(list(self.__model__.beta),
                                                            list(self.__model__.beta[:, :].value))])
        beta = pd.DataFrame(beta, columns=['Name', 'Key', 'Value'])
        beta = beta.pivot(index='Name', columns='Key', values='Value')
        return beta.to_numpy()

    def get_residual(self):
        """Return residual value by array"""
        if self.optimization_status == 0:
            print("Model isn't optimized. Use optimize() method to estimate the model.")
            return False
        residual = list(self.__model__.epsilon[:].value)
        return np.asarray(residual)

    def get_positive_residual(self):
        """Return positive residual value by array"""
        if self.optimization_status == 0:
            print("Model isn't optimized. Use optimize() method to estimate the model.")
            return False
        residual_plus = list(self.__model__.epsilon_plus[:].value)
        return np.asarray(residual_plus)

    def get_negative_residual(self):
        """Return negative residual value by array"""
        if self.optimization_status == 0:
            print("Model isn't optimized. Use optimize() method to estimate the model.")
            return False
        residual_minus = list(self.__model__.epsilon_minus[:].value)
        return np.asarray(residual_minus)

    def get_frontier(self):
        """Return estimated frontier value by array"""
        if self.optimization_status == 0:
            print("Model isn't optimized. Use optimize() method to estimate the model.")
            return False
        if self.cet == "mult":
            frontier = np.asarray(list(self.__model__.frontier[:].value))+1
        elif self.cet == "addi":
            frontier = np.asarray(self.y) - self.get_residual()
        return np.asarray(frontier)

    def plot2d(self, xselect, fig_name=None):
        """Plot with selected x"""
        x = np.array(self.x).T[xselect]
        y = np.array(self.y).T
        f = y - self.get_residual()
        data = (np.stack([x, y, f], axis=0)).T

        # sort
        data = data[np.argsort(data[:, 0])].T

        x, y, f = data[0], data[1], data[2]

        # create figure and axes objects
        fig, ax = plt.subplots()
        dp = ax.scatter(x, y, color="k", marker='x')
        fl = ax.plot(x, f, color="r", label="CNLS")

        # add legend
        legend = plt.legend([dp, fl[0]],
                            ['Data points', 'CNLS'],
                            loc='upper left',
                            ncol=1,
                            fontsize=10,
                            frameon=False)

        # add x, y label
        ax.set_xlabel("Input $x$%d" % (xselect))
        ax.set_ylabel("Output $y$")

        # Remove top and right axes
        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)
        if fig_name == None:
            plt.show()
        else:
            plt.savefig(fig_name)


class CER(CQR):
    """Convex expectile regression (CER)"""

    def __init__(self, y, x, tau, z=None, cet='addi', fun='prod', rts='vrs'):
        """
           * y : Output variable
           * x : Input variables
           * z: Contextual variables  
           * tau : expectile
           * cet  = "addi" : Additive composite error term
                  = "mult" : Multiplicative composite error term
           * fun  = "prod" : Production frontier
                  = "cost" : Cost frontier
           * rts  = "vrs"  : Variable returns to scale
                  = "crs"  : Constant returns to scale
        """
        super().__init__(y, x, tau, z, cet, fun, rts)
        self.__model__.objective.deactivate()
        self.__model__.squared_objective = Objective(
            rule=self.__squared_objective_rule(), sense=minimize, doc='squared objective rule')

    def __squared_objective_rule(self):
        def squared_objective_rule(model):
            return self.tau * sum(model.epsilon_plus[i] ** 2 for i in model.I)\
                + (1 - self.tau) * \
                sum(model.epsilon_minus[i] ** 2 for i in model.I)
        return squared_objective_rule
