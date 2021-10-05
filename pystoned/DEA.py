# import dependencies
from pyomo.environ import ConcreteModel, Set, Var, Objective, minimize, maximize, Constraint
import numpy as np
import pandas as pd
from .constant import CET_ADDI, ORIENT_IO, ORIENT_OO, RTS_VRS, RTS_CRS, OPT_DEFAULT, OPT_LOCAL
from .utils import tools


class DEA:
    """Data Envelopment Analysis (DEA)
    """

    def __init__(self, y, x, orient, rts, yref=None, xref=None):
        """DEA: Envelopment problem 

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

        if type(yref) != type(None):
            self.yref, self.xref = tools.assert_valid_reference_data(
                self.y, self.x, yref, xref)
        else:
            self.yref, self.xref = self.y, self.x
        self.__model__.R = Set(initialize=range(len(self.yref)))

        # Initialize sets
        self.__model__.I = Set(initialize=range(len(self.y)))
        self.__model__.J = Set(initialize=range(len(self.x[0])))
        self.__model__.K = Set(initialize=range(len(self.y[0])))

        # Initialize variable
        self.__model__.theta = Var(self.__model__.I, doc='efficiency')
        self.__model__.lamda = Var(self.__model__.I, self.__model__.R, bounds=(
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

    def __objective_rule(self):
        """Return the proper objective function"""
        def objective_rule(model):
            return sum(model.theta[i] for i in model.I)
        return objective_rule

    def __input_rule(self):
        """Return the proper input constraint"""
        if self.orient == ORIENT_IO:
            def input_rule(model, o, j):
                return model.theta[o]*self.x[o][j] >= sum(model.lamda[o, r]*self.xref[r][j] for r in model.R)
            return input_rule
        elif self.orient == ORIENT_OO:
            def input_rule(model, o, j):
                return sum(model.lamda[o, r] * self.xref[r][j] for r in model.R) <= self.x[o][j]
            return input_rule

    def __output_rule(self):
        """Return the proper output constraint"""
        if self.orient == ORIENT_IO:
            def output_rule(model, o, k):
                return sum(model.lamda[o, r] * self.yref[r][k] for r in model.R) >= self.y[o][k]
            return output_rule
        elif self.orient == ORIENT_OO:
            def output_rule(model, o, k):
                return model.theta[o]*self.y[o][k] <= sum(model.lamda[o, r]*self.yref[r][k] for r in model.R)
            return output_rule

    def __vrs_rule(self):
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
        lamda = np.asarray([i + tuple([j]) for i, j in zip(list(self.__model__.lamda),
                                                          list(self.__model__.lamda[:, :].value))])
        lamda = pd.DataFrame(lamda, columns=['Name', 'Key', 'Value'])
        lamda = lamda.pivot(index='Name', columns='Key', values='Value')
        return lamda.to_numpy()


class DDF(DEA):
    def __init__(self,  y, x, b=None, gy=[1], gx=[1], gb=None, rts=RTS_VRS, yref=None, xref=None, bref=None):
        """DEA: Directional distance function

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

        self.y, self.x, self.b, self.gy, self.gx, self.gb = tools.assert_valid_direciontal_data(
            y, x, b, gy, gx, gb)
        self.rts = rts

        if type(yref) != type(None):
            self.yref, self.xref, self.bref = tools.assert_valid_reference_data_with_bad_outputs(
                self.y, self.x, self.b, yref, xref, bref)
        else:
            self.yref, self.xref, self.bref = self.y, self.x, self.b
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

        self.__model__.lamda = Var(self.__model__.I, self.__model__.R, bounds=(
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
        def input_rule(model, o, j):
            return self.x[o][j] - model.theta[o]*self.gx[j] >= sum(model.lamda[o, r] * self.xref[r][j] for r in model.R)
        return input_rule

    def __output_rule(self):
        """Return the proper output constraint"""
        def output_rule(model, o, k):
            return self.y[o][k] + model.theta[o]*self.gy[k] <= sum(model.lamda[o, r] * self.yref[r][k] for r in model.R)
        return output_rule

    def __undesirable_output_rule(self):
        """Return the proper undesirable output constraint"""
        def undesirable_output_rule(model, o, l):
            return self.b[o][l] - model.theta[o]*self.gb[l] == sum(model.lamda[o, r] * self.bref[r][l] for r in model.R)
        return undesirable_output_rule

    def __vrs_rule(self):
        """Return the VRS constraint"""
        def vrs_rule(model, o):
            return sum(model.lamda[o, r] for r in model.R) == 1
        return vrs_rule


class DUAL(DEA):
    def __init__(self, y, x, orient, rts, yref=None, xref=None):
        """DEA: Multiplier problem

        Args:
            y (float): output variable. 
            x (float): input variables.
            orient (String): ORIENT_IO (input orientation) or ORIENT_OO (output orientation)
            rts (String): RTS_VRS (variable returns to scale) or RTS_CRS (constant returns to scale)
            yref (String, optional): reference output. Defaults to None.
            xref (String, optional): reference inputs. Defaults to None.
        """
        # Initialize DEA model
        self.__model__ = ConcreteModel()

        self.y, self.x = tools.assert_valid_mupltiple_y_data(y, x)
        self.orient = orient
        self.rts = rts

        if type(yref) != type(None):
            self.yref, self.xref = tools.assert_valid_reference_data(
                self.y, self.x, yref, xref)
        else:
            self.yref, self.xref = self.y, self.x
        self.__model__.R = Set(initialize=range(len(self.yref)))

        # Initialize sets
        self.__model__.I = Set(initialize=range(len(self.y)))
        self.__model__.J = Set(initialize=range(len(self.x[0])))
        self.__model__.K = Set(initialize=range(len(self.y[0])))

        # Initialize variable
        self.__model__.nu = Var(self.__model__.I, self.__model__.J, bounds=(
            0.0, None), doc='multiplier x')
        self.__model__.mu = Var(self.__model__.I, self.__model__.K, bounds=(
            0.0, None), doc='multiplier y')
        if self.rts == RTS_VRS:
            self.__model__.omega = Var(self.__model__.I, doc='variable return to scale')

        # Setup the objective function and constraints
        if self.orient == ORIENT_IO:
            self.__model__.objective = Objective(
                rule=self.__objective_rule(), sense=maximize, doc='objective function')
        else:
            self.__model__.objective = Objective(
                rule=self.__objective_rule(), sense=minimize, doc='objective function')
        self.__model__.first = Constraint(
            self.__model__.I, self.__model__.R, rule=self.__first_rule(), doc='technology constraint')
        self.__model__.second = Constraint(
            self.__model__.I, rule=self.__second_rule(), doc='normalization constraint')

        # Optimize model
        self.optimization_status = 0
        self.problem_status = 0

    def __objective_rule(self):
        """Return the proper objective function"""
        if self.orient == ORIENT_IO:
            def objective_rule(model):
                if self.rts == RTS_VRS:
                    return sum(sum(model.mu[o, k] * self.y[o][k] for o in model.I) for k in model.K) + sum(model.omega[o] for o in model.I)
                elif self.rts == RTS_CRS: 
                    return sum(sum(model.mu[o, k] * self.y[o][k] for o in model.I) for k in model.K)
            return objective_rule
        elif self.orient == ORIENT_OO:
            def objective_rule(model):
                if self.rts == RTS_VRS:
                    return sum(sum(model.nu[o, j] * self.x[o][j] for o in model.I) for j in model.J) + sum(model.omega[o] for o in model.I)
                elif self.rts == RTS_CRS: 
                    return sum(sum(model.nu[o, j] * self.x[o][j] for o in model.I) for j in model.J)
            return objective_rule            
 
    def __first_rule(self):
        """Return the proper technology constraint"""
        if self.orient == ORIENT_IO:
            if self.rts == RTS_VRS:
                def first_rule(model, o, r):
                    return sum(model.mu[o, k] * self.yref[r][k] for k in model.K) - sum(model.nu[o, j] * self.xref[r][j] for j in model.J) + model.omega[o] <= 0
                return first_rule
            elif self.rts == RTS_CRS:
                def first_rule(model, o, r):
                    return sum(model.mu[o, k] * self.yref[r][k] for k in model.K) - sum(model.nu[o, j] * self.xref[r][j] for j in model.J)  <= 0
                return first_rule         
        elif self.orient == ORIENT_OO:
            if self.rts == RTS_VRS:
                def first_rule(model, o, r):
                    return sum(model.nu[o, j] * self.xref[r][j] for j in model.J) - sum(model.mu[o, k] * self.yref[r][k] for k in model.K) + model.omega[o] >= 0
                return first_rule
            elif self.rts == RTS_CRS:
                def first_rule(model, o, r):
                    return sum(model.nu[o, j] * self.xref[r][j] for j in model.J) - sum(model.mu[o, k] * self.yref[r][k] for k in model.K) >= 0
                return first_rule      

    def __second_rule(self):
        """Return the proper normalization constraint"""
        if self.orient == ORIENT_IO:
            def second_rule(model, o):
                return sum(model.nu[o, j] * self.x[o][j] for j in model.J) == 1
            return second_rule
        elif self.orient == ORIENT_OO:
            def second_rule(model, o):
                return sum(model.mu[o, k] * self.y[o][k] for k in model.K) == 1
            return second_rule

    def display_mu(self):
        """Display mu value"""
        tools.assert_optimized(self.optimization_status)
        self.__model__.mu.display()

    def display_nu(self):
        """Display nu value"""
        tools.assert_optimized(self.optimization_status)
        self.__model__.nu.display()

    def display_omega(self):
         """Display omega value"""
         tools.assert_optimized(self.optimization_status)
         tools.assert_various_return_to_scale_omega(self.rts)
         self.__model__.omega.display()       

    def get_mu(self):
        """Return mu value by array"""
        tools.assert_optimized(self.optimization_status)
        mu = np.asarray([i + tuple([j]) for i, j in zip(list(self.__model__.mu),
                                                          list(self.__model__.mu[:, :].value))])
        mu = pd.DataFrame(mu, columns=['Name', 'Key', 'Value'])
        mu = mu.pivot(index='Name', columns='Key', values='Value')
        return mu.to_numpy()

    def get_nu(self):
        """Return nu value by array"""
        tools.assert_optimized(self.optimization_status)
        nu = np.asarray([i + tuple([j]) for i, j in zip(list(self.__model__.nu),
                                                          list(self.__model__.nu[:, :].value))])
        nu = pd.DataFrame(nu, columns=['Name', 'Key', 'Value'])
        nu = nu.pivot(index='Name', columns='Key', values='Value')
        return nu.to_numpy()

    def get_omega(self):
        """Return omega value by array"""
        tools.assert_optimized(self.optimization_status)
        tools.assert_various_return_to_scale_omega(self.rts)
        omega = list(self.__model__.omega[:].value)
        return np.asarray(omega)

    def get_efficiency(self):
        """Return efficiency value by array"""
        tools.assert_optimized(self.optimization_status)
        if self.orient == ORIENT_IO:
            if self.rts == RTS_CRS:
                return (np.sum(self.get_mu()*self.y, axis=1)).reshape(len(self.y), 1)
            elif self.rts == RTS_VRS:
                return (np.sum(self.get_mu()*self.y, axis=1)).reshape(len(self.y), 1) + self.get_omega().reshape(len(self.y), 1)
        elif self.orient == ORIENT_OO:
            if self.rts == RTS_CRS:
                return (np.sum(self.get_nu()*self.x, axis=1)).reshape(len(self.x), 1)
            elif self.rts == RTS_VRS:
                return (np.sum(self.get_nu()*self.x, axis=1)).reshape(len(self.x), 1) + self.get_omega().reshape(len(self.x), 1)
                