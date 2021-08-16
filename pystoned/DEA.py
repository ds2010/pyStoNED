# import dependencies
from pyomo.environ import ConcreteModel, Set, Var, Objective, minimize, maximize, Constraint
import numpy as np

from .constant import CET_ADDI, ORIENT_IO, ORIENT_OO, RTS_VRS, OPT_DEFAULT, OPT_LOCAL
from .utils import tools


class DEA:
    """Data Envelopment Analysis (DEA)
    """

    def __init__(self, y, x, orient, rts, yref=None, xref=None):
        """DEA model

        Args:
            y (float): output variable. 
            x (float): input variables.
            orient (String): ORIENT_IO (input orientation) or ORIENT_OO (output orientation)
            rts (String): RTS_VRS (variable returns to scale) or RTS_CRS (constant returns to scale)
            yref (String, optional): reference output. Defaults to None.
            xref (String, optional): reference inputs. Defaults to None.
        """
        # TODO(error/warning handling): Check the configuration of the model exist
        # Initialize DEA model
        self.__model__ = ConcreteModel()

        self.y, self.x = tools.assert_valid_mupltiple_y_data(y, x)
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
        if self.orient == ORIENT_IO:
            self.__model__.objective = Objective(
                rule=self.__objective_rule(), sense=minimize, doc='objective function')
        else:
            self.__model__.objective = Objective(
                rule=self.__objective_rule(), sense=maximize, doc='objective function')
        self.__model__.input = Constraint(
            self.__model__.I, self.__model__.J, rule=self.__input_rule(), doc='input constraint')
        self.__model__.output = Constraint(
            self.__model__.I, self.__model__.K, rule=self.__output_rule(), doc='output constraint')
        if self.rts == RTS_VRS:
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
            if self.orient == ORIENT_IO:
                def input_rule(model, o, j):
                    return model.theta[o]*self.x[o][j] >= sum(model.lamda[o, i]*self.x[i][j] for i in model.I)
                return input_rule
            elif self.orient == ORIENT_OO:
                def input_rule(model, o, j):
                    return sum(model.lamda[o, i] * self.x[i][j] for i in model.I) <= self.x[o][j]
                return input_rule
        else:
            if self.orient == ORIENT_IO:
                def input_rule(model, o, j):
                    return model.theta[o]*self.x[o][j] >= sum(model.lamda[o, r]*self.xref[r][j] for r in model.R)
                return input_rule
            elif self.orient == ORIENT_OO:
                def input_rule(model, o, j):
                    return sum(model.lamda[o, r] * self.x[r][j] for r in model.I) <= self.x[o][j]
                return input_rule

    def __output_rule(self):
        """Return the proper output constraint"""
        if self.__reference == False:
            if self.orient == ORIENT_IO:
                def output_rule(model, o, k):
                    return sum(model.lamda[o, i] * self.y[i][k] for i in model.I) >= self.y[o][k]
                return output_rule
            elif self.orient == ORIENT_OO:
                def output_rule(model, o, k):
                    return model.theta[o]*self.y[o][k] <= sum(model.lamda[o, i]*self.y[i][k] for i in model.I)
                return output_rule
        else:
            if self.orient == ORIENT_IO:
                def output_rule(model, o, k):
                    return sum(model.lamda[o, r] * self.yref[r][k] for r in model.R) >= self.y[o][k]
                return output_rule
            elif self.orient == ORIENT_OO:
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

    def optimize(self, email=OPT_LOCAL, solver=OPT_DEFAULT):
        """Optimize the function by requested method

        Args:
            email (string): The email address for remote optimization. It will optimize locally if OPT_LOCAL is given.
            solver (string): The solver chosen for optimization. It will optimize with default solver if OPT_DEFAULT is given.
        """
        # TODO(error/warning handling): Check problem status after optimization
        self.problem_status, self.optimization_status = tools.optimize_model(
            self.__model__, email, CET_ADDI, solver)

    def display_status(self):
        """Display the status of problem"""
        print(self.optimization_status)

    def display_theta(self):
        """Display theta value"""
        tools.assert_optimized(self.optimization_status)
        self.__model__.theta.display()

    def display_lamda(self):
        """Display lamda value"""
        tools.assert_optimized(self.optimization_status)
        self.__model__.lamda.display()

    def get_status(self):
        """Return status"""
        return self.optimization_status

    def get_theta(self):
        """Return theta value by array"""
        tools.assert_optimized(self.optimization_status)
        theta = list(self.__model__.theta[:].value)
        return np.asarray(theta)

    def get_lamda(self):
        """Return lamda value by array"""
        tools.assert_optimized(self.optimization_status)
        lamda = list(self.__model__.lamda[:].value)
        return np.asarray(lamda)


class DEADDF(DEA):
    def __init__(self,  y, x, b=None, gy=[1], gx=[1], gb=None, rts=RTS_VRS, yref=None, xref=None, bref=None):
        """DEA DDF model 

        Args:
            y (float): output variable. 
            x (float): input variables.
            b (float), optional): undesirable output variables. Defaults to None.
            gy (list, optional): output directional vector. Defaults to [1].
            gx (list, optional): input directional vector. Defaults to [1].
            gb (list, optional): undesirable output directional vector. Defaults to None.
            rts (String): RTS_VRS (variable returns to scale) or RTS_CRS (constant returns to scale)
            yref (String, optional): reference output. Defaults to None.
            xref (String, optional): reference inputs. Defaults to None.
            bref (String, optional): reference undesirable output. Defaults to None.
        """
        # Initialize DEA model
        self.__model__ = ConcreteModel()

        self.y, self.x, self.b, self.gy, self.gx, self.gb = tools.assert_valid_direciontal_data(y,x,b,gy,gx,gb)
        self.rts = rts

        self.__reference = False

        if type(yref) != type(None):
            self.__reference = True
            self.yref = self._DEA__to_2d_list(yref)
            self.xref = self._DEA__to_2d_list(xref)
            if type(b) != type(None):
                self.bref = self._DEA__to_2d_list(bref)
            self.__model__.R = Set(initialize=range(len(self.yref)))

        # Initialize sets
        self.__model__.I = Set(initialize=range(len(self.y)))
        self.__model__.J = Set(initialize=range(len(self.x[0])))
        self.__model__.K = Set(initialize=range(len(self.y[0])))
        if type(b) != type(None):
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

        if type(b) != type(None):
            self.__model__.undesirable_output = Constraint(
                self.__model__.I, self.__model__.L, rule=self.__undesirable_output_rule(), doc='undesirable output constraint')

        if self.rts == RTS_VRS:
            self.__model__.vrs = Constraint(
                self.__model__.I, rule=self.__vrs_rule(), doc='various return to scale rule')

        # Optimize model
        self.optimization_status = 0
        self.problem_status = 0

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
