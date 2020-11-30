# Import pyomo module
from pyomo.environ import ConcreteModel, Set, Var, Objective, minimize, maximize, Constraint, Binary
from pyomo.opt import SolverFactory, SolverManagerFactory
from pyomo.core.expr.numvalue import NumericValue
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class FDH:
    """
    Free Disposal Hull (FDH)
    """

    def __init__(self, y, x, orient):
        """
        orient  = "io" : input orientation
                = "oo" : output orientation
        """
        # TODO(error/warning handling): Check the configuration of the model exist
        self.x = x.tolist()
        self.y = y.tolist()
        self.orient = orient

        if type(self.x[0]) != list:
            self.x = []
            for x_value in x.tolist():
                self.x.append([x_value])

        if type(self.y[0]) != list:
            self.y = []
            for y_value in y.tolist():
                self.y.append([y_value])

        # Initialize DEA model
        self.__model__ = ConcreteModel()

        # Initialize sets
        self.__model__.I = Set(initialize=range(len(self.y)))
        self.__model__.J = Set(initialize=range(len(self.x[0])))
        self.__model__.K = Set(initialize=range(len(self.y[0])))

        # Initialize variable
        self.__model__.theta = Var(self.__model__.I, doc='efficiency')
        self.__model__.lamda = Var(
            self.__model__.I, self.__model__.I, within=Binary, doc='intensity variables')

        # Setup the objective function and constraints
        if self.orient == "io":
            self.__model__.objective = Objective(
                rule=self.__objective_rule(), sense=minimize, doc='objective function')
        else:
            self.__model__.objective = Objective(
                rule=self.__objective_rule(), sense=maximize, doc='objective function')
        self.__model__.input = Constraint(
            self.__model__.I, self.__model__.J, rule=self.__input_rule(), doc='input constraint')
        self.__model__.output = Constraint(
            self.__model__.I, self.__model__.K, rule=self.__output_rule(), doc='output constraint')
        self.__model__.vrs = Constraint(
            self.__model__.I, rule=self.__vrs_rule(), doc='various return to scale rule')

        # Optimize model
        self.optimization_status = 0
        self.problem_status = 0

    def __objective_rule(self):
        """Return the proper objective function"""

        def objective_rule(model):
            return sum(model.theta[i] for i in model.I)

        return objective_rule

    def __input_rule(self):
        """Return the proper input constraint"""
        if self.orient == "io":
            def input_rule(model, o, j):
                return model.theta[o]*self.x[o][j] >= sum(model.lamda[o, i]*self.x[i][j] for i in model.I)
            return input_rule
        elif self.orient == "oo":
            def input_rule(model, o, j):
                return sum(model.lamda[o, i] * self.x[i][j] for i in model.I) <= self.x[o][j]
            return input_rule

    def __output_rule(self):
        """Return the proper output constraint"""
        if self.orient == "io":
            def output_rule(model, o, k):
                return sum(model.lamda[o, i] * self.y[i][k] for i in model.I) >= self.y[o][k]
            return output_rule
        elif self.orient == "oo":
            def output_rule(model, o, k):
                return model.theta[o]*self.y[o][k] <= sum(model.lamda[o, i]*self.y[i][k] for i in model.I)
            return output_rule

    def __vrs_rule(self):
        def vrs_rule(model, o):
            return sum(model.lamda[o, i] for i in model.I) == 1
        return vrs_rule

    def optimize(self, remote=True):
        """Optimize the function by requested method"""
        if remote == False:
            solver = SolverFactory("mosek")
            print("Estimating the model locally with mosek solver")
            self.problem_status = solver.solve(self.__model__, tee=True)
            self.optimization_status = 1
        else:
            solver = SolverManagerFactory("neos")
            print("Estimating the model remotely with mosek solver")
            self.problem_status = solver.solve(
                self.__model__, tee=True, opt="mosek")
            self.optimization_status = 1

    def display_status(self):
        """Display the status of problem"""
        if self.optimization_status == 0:
            print("Model isn't optimized. Use optimize() method to estimate the model.")
            return False
        print(self.display_status)

    def display_theta(self):
        """Display theta value"""
        if self.optimization_status == 0:
            print("Model isn't optimized. Use optimize() method to estimate the model.")
            return False
        self.__model__.theta.display()

    def display_lamda(self):
        """Display lamda value"""
        if self.optimization_status == 0:
            print("Model isn't optimized. Use optimize() method to estimate the model.")
            return False
        self.__model__.lamda.display()

    def get_status(self):
        """Return status"""
        if self.optimization_status == 0:
            print("Model isn't optimized. Use optimize() method to estimate the model.")
            return False
        return self.optimization_status

    def get_theta(self):
        """Return theta value by array"""
        if self.optimization_status == 0:
            print("Model isn't optimized. Use optimize() method to estimate the model.")
            return False
        theta = list(self.__model__.theta[:].value)
        return np.asarray(theta)

    def get_lamda(self):
        """Return lamda value by array"""
        if self.optimization_status == 0:
            print("Model isn't optimized. Use optimize() method to estimate the model.")
            return False
        lamda = list(self.__model__.lamda[:].value)
        return np.asarray(lamda)
