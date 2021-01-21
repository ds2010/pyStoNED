# Import pyomo module
from pyomo.environ import ConcreteModel, Set, Var, Objective, minimize, Constraint, log
from pyomo.opt import SolverFactory, SolverManagerFactory
from pyomo.core.expr.numvalue import NumericValue
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from .utils import CQERG1, CQERG2, CQERZG1, CQERZG2, sweet


class CQRG:
    """Convex quantile regression (CQR) with Genetic algorithm"""

    def __init__(self, y, x, tau, z=None, cet='addi', fun='prod', rts='vrs'):
        """
        Initialize the CNLSG model

        * y : Output variable
        * x : Input variables
        * z : Contextual variables
        * tau : quantile
        * cet  = "addi" : Additive composite error term
               = "mult" : Multiplicative composite error term
        * fun  = "prod" : Production frontier
               = "cost" : Cost frontier
        * rts  = "vrs"  : Variable returns to scale
               = "crs"  : Constant returns to scale
        """

        # TODO(error/warning handling): Check the configuration of the model exist
        self.cutactive = sweet.sweet(x)
        self.x = x.tolist()
        self.y = y.tolist()
        self.tau = tau
        self.z = z
        self.cet = cet
        self.fun = fun
        self.rts = rts

        # active (added) violated concavity constraint by iterative procedure
        self.Active = np.zeros((len(x), len(x)))
        # violated concavity constraint
        self.Active2 = np.zeros((len(x), len(x)))

        if type(self.y[0]) == list:
            self.y = self.__to_1d_list(self.y)

        if type(self.x[0]) != list:
            self.x = []
            for x_value in x.tolist():
                self.x.append([x_value])

        if type(self.z) != type(None):
            self.z = z.tolist()
            if type(self.z[0]) != list:
                self.z = []
                for z_value in z.tolist():
                    self.z.append([z_value])
            model1 = CQERZG1.CQRZG1(
                self.y, self.x, self.z, self.tau, self.cutactive, self.cet, self.fun, self.rts)
        else:
            model1 = CQERG1.CQRG1(
                self.y, self.x, self.tau, self.cutactive, self.cet, self.fun, self.rts)
        model1.optimize(remote=False)
        self.alpha = model1.get_alpha()
        self.beta = model1.get_beta()
        self.__modol__ = model1.__model__

        # Optimize model
        self.optimization_status = 0
        self.problem_status = 0

    def optimize(self):
        """Optimize the function by requested method"""
        # TODO(error/warning handling): Check problem status after optimization
        while self.__convergence_test(self.alpha, self.beta) > 0.0001:
            if type(self.z) != type(None):
                model2 = CQERZG2.CQRZG2(
                    self.y, self.x, self.z, self.tau, self.Active, self.cutactive, self.cet, self.fun, self.rts)
            else:
                model2 = CQERG2.CQRG2(
                    self.y, self.x, self.tau, self.Active, self.cutactive, self.cet, self.fun, self.rts)
            model2.optimize(remote=False)
            self.alpha = model2.get_alpha()
            self.beta = model2.get_beta()
            # TODO: Replace print with log system
            print("Genetic Algorithm Convergence : %8f" %
                  (self.__convergence_test(self.alpha, self.beta)))
        self.__model__ = model2.__model__
        self.optimization_status = 1

    def __to_1d_list(self, l):
        rl = []
        for i in range(len(l)):
            rl.append(l[i][0])
        return rl

    def __convergence_test(self, alpha, beta):
        x = np.asarray(self.x)
        Activetmp1 = 0.0

        # go into the loop
        for i in range(len(x)):
            Activetmp = 0.0
            # go into the sub-loop and find the violated concavity constraints
            for j in range(len(x)):
                if self.cet == "addi":
                    if self.rts == "vrs":
                        if self.fun == "prod":
                            self.Active2[i, j] = alpha[i] + np.sum(beta[i, :] * x[i, :]) - \
                                alpha[j] - np.sum(beta[j, :] * x[i, :])
                        elif self.fun == "cost":
                            self.Active2[i, j] = - alpha[i] - np.sum(beta[i, :] * x[i, :]) + \
                                alpha[j] + np.sum(beta[j, :] * x[i, :])
                if self.cet == "mult":
                    if self.rts == "vrs":
                        if self.fun == "prod":
                            self.Active2[i, j] = alpha[i] + np.sum(beta[i, :] * x[i, :]) - \
                                alpha[j] - np.sum(beta[j, :] * x[i, :])
                        elif self.fun == "cost":
                            self.Active2[i, j] = - alpha[i] - np.sum(beta[i, :] * x[i, :]) + \
                                alpha[j] + np.sum(beta[j, :] * x[i, :])
                    if self.rts == "crs":
                        if self.fun == "prod":
                            self.Active2[i, j] = np.sum(beta[i, :] * x[i, :]) - \
                                np.sum(beta[j, :] * x[i, :])
                        elif self.fun == "cost":
                            self.Active2[i, j] = - np.sum(beta[i, :] * x[i, :]) + \
                                np.sum(beta[j, :] * x[i, :])
                if self.Active2[i, j] > Activetmp:
                    Activetmp = self.Active2[i, j]
            # find the maximal violated constraint in sub-loop and added into the active matrix
            for j in range(len(x)):
                if self.Active2[i, j] >= Activetmp and Activetmp > 0:
                    self.Active[i, j] = 1
            if Activetmp > Activetmp1:
                Activetmp1 = Activetmp
        return Activetmp

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

    def display_lamda(self):
        """Display lamda value"""
        if self.optimization_status == 0:
            print("Model isn't optimized. Use optimize() method to estimate the model.")
            return False
        if type(self.z) == type(None):
            # TODO: Replace print by warning
            print("Without z variable")
            return
        self.__model__.lamda.display()

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

    def get_lamda(self):
        """Return beta value by array"""
        if self.optimization_status == 0:
            print("Model isn't optimized. Use optimize() method to estimate the model.")
            return False
        if type(self.z) == type(None):
            # TODO: Replace print by warning
            print("Without z variable")
            return
        lamda = list(self.__model__.lamda[:].value)
        return np.asarray(lamda)

    def get_frontier(self):
        """Return estimated frontier value by array"""
        if self.optimization_status == 0:
            print("Model isn't optimized. Use optimize() method to estimate the model.")
            return False
        if self.cet == "mult" and type(self.z) == type(None):
            frontier = np.asarray(list(self.__model__.frontier[:].value))+1
        elif self.cet == "mult" and type(self.z) != type(None):
            frontier = list(np.divide(self.y, np.exp(
                self.get_residual()+self.get_lamda()*np.asarray(self.z)[:, 0])) - 1)
        elif self.cet == "addi":
            frontier = np.asarray(self.y) - self.get_residual()
        return np.asarray(frontier)


class CERG:
    """Convex expectile regression (CER) with Genetic algorithm"""

    def __init__(self, y, x, tau, z=None, cet='addi', fun='prod', rts='vrs'):
        """
        Initialize the CNLSG model

        * y : Output variable
        * x : Input variables
        * z : Contextual variables
        * tau : expectile
        * cet  = "addi" : Additive composite error term
               = "mult" : Multiplicative composite error term
        * fun  = "prod" : Production frontier
               = "cost" : Cost frontier
        * rts  = "vrs"  : Variable returns to scale
               = "crs"  : Constant returns to scale
        """

        # TODO(error/warning handling): Check the configuration of the model exist
        self.cutactive = sweet.sweet(x)
        self.x = x.tolist()
        self.y = y.tolist()
        self.tau = tau
        self.z = z
        self.cet = cet
        self.fun = fun
        self.rts = rts

        # active (added) violated concavity constraint by iterative procedure
        self.Active = np.zeros((len(x), len(x)))
        # violated concavity constraint
        self.Active2 = np.zeros((len(x), len(x)))

        if type(self.y[0]) == list:
            self.y = self.__to_1d_list(self.y)

        if type(self.x[0]) != list:
            self.x = []
            for x_value in x.tolist():
                self.x.append([x_value])

        if type(self.z) != type(None):
            self.z = z.tolist()
            if type(self.z[0]) != list:
                self.z = []
                for z_value in z.tolist():
                    self.z.append([z_value])
            model1 = CQERZG1.CERZG1(
                self.y, self.x, self.z, self.tau, self.cutactive, self.cet, self.fun, self.rts)
        else:
            model1 = CQERG1.CERG1(
                self.y, self.x, self.tau, self.cutactive, self.cet, self.fun, self.rts)
        model1.optimize(remote=False)
        self.alpha = model1.get_alpha()
        self.beta = model1.get_beta()
        self.__modol__ = model1.__model__

        # Optimize model
        self.optimization_status = 0
        self.problem_status = 0

    def optimize(self):
        """Optimize the function by requested method"""
        # TODO(error/warning handling): Check problem status after optimization
        while self.__convergence_test(self.alpha, self.beta) > 0.0001:
            if type(self.z) != type(None):
                model2 = CQERZG2.CERZG2(
                    self.y, self.x, self.z, self.tau, self.Active, self.cutactive, self.cet, self.fun, self.rts)
            else:
                model2 = CQERG2.CERG2(
                    self.y, self.x, self.tau, self.Active, self.cutactive, self.cet, self.fun, self.rts)
            model2.optimize(remote=False)
            self.alpha = model2.get_alpha()
            self.beta = model2.get_beta()
            # TODO: Replace print with log system
            print("Genetic Algorithm Convergence : %8f" %
                  (self.__convergence_test(self.alpha, self.beta)))
        self.__model__ = model2.__model__
        self.optimization_status = 1

    def __to_1d_list(self, l):
        rl = []
        for i in range(len(l)):
            rl.append(l[i][0])
        return rl

    def __convergence_test(self, alpha, beta):
        x = np.asarray(self.x)
        Activetmp1 = 0.0

        # go into the loop
        for i in range(len(x)):
            Activetmp = 0.0
            # go into the sub-loop and find the violated concavity constraints
            for j in range(len(x)):
                if self.cet == "addi":
                    if self.rts == "vrs":
                        if self.fun == "prod":
                            self.Active2[i, j] = alpha[i] + np.sum(beta[i, :] * x[i, :]) - \
                                alpha[j] - np.sum(beta[j, :] * x[i, :])
                        elif self.fun == "cost":
                            self.Active2[i, j] = - alpha[i] - np.sum(beta[i, :] * x[i, :]) + \
                                alpha[j] + np.sum(beta[j, :] * x[i, :])
                if self.cet == "mult":
                    if self.rts == "vrs":
                        if self.fun == "prod":
                            self.Active2[i, j] = alpha[i] + np.sum(beta[i, :] * x[i, :]) - \
                                alpha[j] - np.sum(beta[j, :] * x[i, :])
                        elif self.fun == "cost":
                            self.Active2[i, j] = - alpha[i] - np.sum(beta[i, :] * x[i, :]) + \
                                alpha[j] + np.sum(beta[j, :] * x[i, :])
                    if self.rts == "crs":
                        if self.fun == "prod":
                            self.Active2[i, j] = np.sum(beta[i, :] * x[i, :]) - \
                                np.sum(beta[j, :] * x[i, :])
                        elif self.fun == "cost":
                            self.Active2[i, j] = - np.sum(beta[i, :] * x[i, :]) + \
                                np.sum(beta[j, :] * x[i, :])
                if self.Active2[i, j] > Activetmp:
                    Activetmp = self.Active2[i, j]
            # find the maximal violated constraint in sub-loop and added into the active matrix
            for j in range(len(x)):
                if self.Active2[i, j] >= Activetmp and Activetmp > 0:
                    self.Active[i, j] = 1
            if Activetmp > Activetmp1:
                Activetmp1 = Activetmp
        return Activetmp

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

    def display_lamda(self):
        """Display lamda value"""
        if self.optimization_status == 0:
            print("Model isn't optimized. Use optimize() method to estimate the model.")
            return False
        if type(self.z) == type(None):
            # TODO: Replace print by warning
            print("Without z variable")
            return
        self.__model__.lamda.display()

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

    def get_lamda(self):
        """Return beta value by array"""
        if self.optimization_status == 0:
            print("Model isn't optimized. Use optimize() method to estimate the model.")
            return False
        if type(self.z) == type(None):
            # TODO: Replace print by warning
            print("Without z variable")
            return
        lamda = list(self.__model__.lamda[:].value)
        return np.asarray(lamda)

    def get_frontier(self):
        """Return estimated frontier value by array"""
        if self.optimization_status == 0:
            print("Model isn't optimized. Use optimize() method to estimate the model.")
            return False
        if self.cet == "mult" and type(self.z) == type(None):
            frontier = np.asarray(list(self.__model__.frontier[:].value))+1
        elif self.cet == "mult" and type(self.z) != type(None):
            frontier = list(np.divide(self.y, np.exp(
                self.get_residual()+self.get_lamda()*np.asarray(self.z)[:, 0])) - 1)
        elif self.cet == "addi":
            frontier = np.asarray(self.y) - self.get_residual()
        return np.asarray(frontier)
