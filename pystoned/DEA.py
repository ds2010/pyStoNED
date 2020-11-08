# Import pyomo module
from pyomo.environ import ConcreteModel, Set, Var, Objective, minimize, maximize, Constraint
from pyomo.opt import SolverFactory, SolverManagerFactory
from pyomo.core.expr.numvalue import NumericValue
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class DEA:
    """
    Data Envelopment Analysis (DEA)
    """

    def __init__(self, y, x, orient, rts, yref=None, xref=None):
        """
        orient  = "io" : input orientation
                = "oo" : output orientation
        rts     = "vrs": variable returns to scale
                = "crs": constant returns to scale
        """
        # TODO(error/warning handling): Check the configuration of the model exist
        # Initialize DEA model
        self.__model__ = ConcreteModel()

        self.x = self.__to_2d_list(x.tolist())
        self.y = self.__to_2d_list(y.tolist())
        self.orient = orient
        self.rts = rts
        self.__reference = False

        if type(yref) != type(None):
            self.__reference = True
            self.yref = self._DEA__to_2d_list(yref)
            self.xref = self._DEA__to_2d_list(xref)
            self.__model__.R = Set(initialize=range(len(self.yref)))

        # Initialize sets
        self.__model__.I = Set(initialize=range(len(self.y)))
        self.__model__.J = Set(initialize=range(len(self.x[0])))
        self.__model__.K = Set(initialize=range(len(self.y[0])))

        # Initialize variable
        self.__model__.theta = Var(self.__model__.I, doc='efficiency')
        self.__model__.lamda = Var(self.__model__.I, self.__model__.I, bounds=(
            0.0, None), doc='intensity variables')

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
        if self.rts == "vrs":
            self.__model__.vrs = Constraint(
                self.__model__.I, rule=self.__vrs_rule(), doc='various return to scale rule')

        # Optimize model
        self.optimization_status = 0
        self.problem_status = 0

    def __to_2d_list(self, l):
        """If the given list l is 1d list, return its 2d adaption"""
        if type(l[0]) != list:
            r = []
            for value in l:
                r.append([value])
            return r
        return l

    def __objective_rule(self):
        """Return the proper objective function"""

        def objective_rule(model):
            return sum(model.theta[i] for i in model.I)

        return objective_rule

    def __input_rule(self):
        """Return the proper input constraint"""
        if self.__reference == False:
            if self.orient == "io":
                def input_rule(model, o, j):
                    return model.theta[o]*self.x[o][j] >= sum(model.lamda[o, i]*self.x[i][j] for i in model.I)
                return input_rule
            elif self.orient == "oo":
                def input_rule(model, o, j):
                    return sum(model.lamda[o, i] * self.x[i][j] for i in model.I) <= self.x[o][j]
                return input_rule
        else:
            if self.orient == "io":
                def input_rule(model, o, j):
                    return model.theta[o]*self.x[o][j] >= sum(model.lamda[o, r]*self.xref[r][j] for r in model.R)
                return input_rule
            elif self.orient == "oo":
                def input_rule(model, o, j):
                    return sum(model.lamda[o, r] * self.x[r][j] for r in model.I) <= self.x[o][j]
                return input_rule

    def __output_rule(self):
        """Return the proper output constraint"""
        if self.__reference == False:
            if self.orient == "io":
                def output_rule(model, o, k):
                    return sum(model.lamda[o, i] * self.y[i][k] for i in model.I) >= self.y[o][k]
                return output_rule
            elif self.orient == "oo":
                def output_rule(model, o, k):
                    return model.theta[o]*self.y[o][k] <= sum(model.lamda[o, i]*self.y[i][k] for i in model.I)
                return output_rule
        else:
            if self.orient == "io":
                def output_rule(model, o, k):
                    return sum(model.lamda[o, r] * self.yref[r][k] for r in model.R) >= self.y[o][k]
                return output_rule
            elif self.orient == "oo":
                def output_rule(model, o, k):
                    return model.theta[o]*self.y[o][k] <= sum(model.lamda[o, r]*self.y[r][k] for r in model.R)
                return output_rule

    def __vrs_rule(self):
        if self.__reference == False:
            def vrs_rule(model, o):
                return sum(model.lamda[o, i] for i in model.I) == 1
            return vrs_rule
        else:
            def vrs_rule(model, o):
                return sum(model.lamda[o, r] for r in model.R) == 1
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


class DEADDF(DEA):
    def __init__(self,  y, x, b=None, gy=[1], gx=[1], gb=None, rts="vrs", yref=None, xref=None, bref=None):
        """
            y : Output variables
            x : Input variables
            b : Undesirable output variables
            gy : Output directional vector
            gx : Input directional vector
            gb : Undesirable output directional vector
            rts     = "vrs": variable returns to scale
                    = "crs": constant returns to scale
            yref : the reference point of y
            xref : the reference point of x
            bref : the reference point of b
        """
        # Initialize DEA model
        self.__model__ = ConcreteModel()

        self.x = self._DEA__to_2d_list(x.tolist())
        self.y = self._DEA__to_2d_list(y.tolist())
        self.rts = rts
        self.gy = self.__to_1d_list(gy)
        self.gx = self.__to_1d_list(gx)
        self.__undesirable_output = False
        self.__reference = False

        if type(b) != type(None):
            self.__undesirable_output = True
            self.b = self._DEA__to_2d_list(b.tolist())
            self.gb = self.__to_1d_list(gb)

        if type(yref) != type(None):
            self.__reference = True
            self.yref = self._DEA__to_2d_list(yref)
            self.xref = self._DEA__to_2d_list(xref)
            if self.__undesirable_output:
                self.bref = self._DEA__to_2d_list(bref)
            self.__model__.R = Set(initialize=range(len(self.yref)))

        # Initialize sets
        self.__model__.I = Set(initialize=range(len(self.y)))
        self.__model__.J = Set(initialize=range(len(self.x[0])))
        self.__model__.K = Set(initialize=range(len(self.y[0])))
        if self.__undesirable_output:
            self.__model__.L = Set(initialize=range(len(self.b[0])))

        # Initialize variable
        self.__model__.theta = Var(
            self.__model__.I, doc='directional distance')
        if self.__reference:
            self.__model__.lamda = Var(self.__model__.I, self.__model__.R, bounds=(
                0.0, None), doc='intensity variables')
        else:
            self.__model__.lamda = Var(self.__model__.I, self.__model__.I, bounds=(
                0.0, None), doc='intensity variables')

        # Setup the objective function and constraints
        self.__model__.objective = Objective(
            rule=self._DEA__objective_rule(), sense=maximize, doc='objective function')
        self.__model__.input = Constraint(
            self.__model__.I, self.__model__.J, rule=self.__input_rule(), doc='input constraint')
        self.__model__.output = Constraint(
            self.__model__.I, self.__model__.K, rule=self.__output_rule(), doc='output constraint')

        if self.__undesirable_output:
            self.__model__.undesirable_output = Constraint(
                self.__model__.I, self.__model__.L, rule=self.__undesirable_output_rule(), doc='undesirable output constraint')

        if self.rts == "vrs":
            self.__model__.vrs = Constraint(
                self.__model__.I, rule=self.__vrs_rule(), doc='various return to scale rule')

        # Optimize model
        self.optimization_status = 0
        self.problem_status = 0

    def __to_1d_list(self, l):
        if type(l) == int or type(l) == float:
            return [l]
        return l

    def __input_rule(self):
        """Return the proper input constraint"""
        if self.__reference == False:
            def input_rule(model, o, j):
                return self.x[o][j] - model.theta[o]*self.gx[j] >= sum(model.lamda[o, i] * self.x[i][j] for i in model.I)
            return input_rule
        else:
            def input_rule(model, o, j):
                return self.x[o][j] - model.theta[o]*self.gx[j] >= sum(model.lamda[o, r] * self.xref[r][j] for r in model.R)
            return input_rule

    def __output_rule(self):
        """Return the proper output constraint"""
        if self.__reference == False:
            def output_rule(model, o, k):
                return self.y[o][k] + model.theta[o]*self.gy[k] <= sum(model.lamda[o, i] * self.y[i][k] for i in model.I)
            return output_rule
        else:
            def output_rule(model, o, k):
                return self.y[o][k] + model.theta[o]*self.gy[k] <= sum(model.lamda[o, r] * self.yref[r][k] for r in model.R)
            return output_rule

    def __undesirable_output_rule(self):
        """Return the proper undesirable output constraint"""
        if self.__reference == False:
            def undesirable_output_rule(model, o, l):
                return self.b[o][l] - model.theta[o]*self.gb[l] == sum(model.lamda[o, i] * self.b[i][l] for i in model.I)
            return undesirable_output_rule
        else:
            def undesirable_output_rule(model, o, l):
                return self.b[o][l] + model.theta[o]*self.gb[l] == sum(model.lamda[o, r] * self.bref[r][l] for r in model.R)
            return undesirable_output_rule

    def __vrs_rule(self):
        if self.__reference == False:
            def vrs_rule(model, o):
                return sum(model.lamda[o, i] for i in model.I) == 1
            return vrs_rule
        else:
            def vrs_rule(model, o):
                return sum(model.lamda[o, r] for r in model.R) == 1
            return vrs_rule
