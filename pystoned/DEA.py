"""
@Title   : Data Envelopment Analysis (DEA)
@Author  : Sheng Dai, Timo Kuosmanen
@Mail    : sheng.dai@aalto.fi (S. Dai); timo.kuosmanen@aalto.fi (T. Kuosmanen)
@Date    : 2020-06-06
"""

# Import of the pyomo module
from pyomo.environ import *
from . import directV
import numpy as np


def dea(y, x, orient, rts):
    # orient  = "io" : input orientation
    #         = "oo" : output orientation
    # rts     = "vrs": variable returns to scale
    #         = "crs": constant returns to scale

    # transform data
    x = x.tolist()
    y = y.tolist()
    
    # number of DMUs
    n = len(y)

    # number of inputs
    if type(x[0]) == int or type(x[0]) == float:
        m = 1
        x = np.asmatrix(np.array(x)).tolist()
    else:
        m = len(x[0])
        x = np.array(x).T.tolist()  

    # number of outputs
    if type(y[0]) == int or type(y[0]) == float:
        p = 1
        y = np.asmatrix(np.array(y)).tolist()
    else:
        p = len(y[0])
        y = np.array(y).T.tolist()  
            
    # Creation of a Concrete Model
    model = ConcreteModel()

    # Set
    model.i = Set(initialize=range(n))
    model.j = Set(initialize=range(m))
    model.k = Set(initialize=range(p))

    # Alias
    model.io = SetOf(model.i)

    # Variables
    model.theta = Var(model.io, doc='efficiency')
    model.lamda = Var(model.io, model.i, bounds=(0.0, None), doc='intensity variables')

    if orient == "io":

        if rts == "crs":

            # objective
            def objective_rule(model):
                return sum(model.theta[io] for io in model.io)

            model.objective = Objective(rule=objective_rule, sense=minimize, doc='objective function')

            # Constraints
            def input_rule(model, io, j):
                arow = x[j]
                return (model.theta[io] * arow[io]) >= sum(model.lamda[io, i] * arow[i] for i in model.i)

            model.input = Constraint(model.io, model.j, rule=input_rule, doc='input constraints')

            def output_rule(model, io, k):
                brow = y[k]
                return sum(model.lamda[io, i] * brow[i] for i in model.i) >= brow[io]

            model.output = Constraint(model.io, model.k, rule=output_rule, doc='output constraints')

        if rts == "vrs":

            # objective
            def objective_rule(model):
                return sum(model.theta[io] for io in model.io)

            model.objective = Objective(rule=objective_rule, sense=minimize, doc='objective function')

            # Constraints
            def input_rule(model, io, j):
                arow = x[j]
                return (model.theta[io] * arow[io]) >= sum(model.lamda[io, i] * arow[i] for i in model.i)

            model.input = Constraint(model.io, model.j, rule=input_rule, doc='input constraints')

            def output_rule(model, io, k):
                brow = y[k]
                return sum(model.lamda[io, i] * brow[i] for i in model.i) >= brow[io]

            model.output = Constraint(model.io, model.k, rule=output_rule, doc='output constraints')

            def vrs_rule(model, io):
                return sum(model.lamda[io, i] for i in model.i) == 1

            model.vrs = Constraint(model.io, rule=vrs_rule, doc='VRS constraints')

    if orient == "oo":

        if rts == "crs":

            # objective
            def objective_rule(model):
                return sum(model.theta[io] for io in model.io)

            model.objective = Objective(rule=objective_rule, sense=maximize, doc='objective function')

            # Constraints
            def input_rule(model, io, j):
                arow = x[j]
                return sum(model.lamda[io, i] * arow[i] for i in model.i) <= arow[io]

            model.input = Constraint(model.io, model.j, rule=input_rule, doc='input constraints')

            def output_rule(model, io, k):
                brow = y[k]
                return model.theta[io] * brow[io] <= sum(model.lamda[io, i] * brow[i] for i in model.i)

            model.output = Constraint(model.io, model.k, rule=output_rule, doc='output constraints')

        if rts == "vrs":

            # objective
            def objective_rule(model):
                return sum(model.theta[io] for io in model.io)

            model.objective = Objective(rule=objective_rule, sense=maximize, doc='objective function')

            # Constraints
            def input_rule(model, io, j):
                arow = x[j]
                return sum(model.lamda[io, i] * arow[i] for i in model.i) <= arow[io]

            model.input = Constraint(model.io, model.j, rule=input_rule, doc='input constraints')

            def output_rule(model, io, k):
                brow = y[k]
                return model.theta[io] * brow[io] <= sum(model.lamda[io, i] * brow[i] for i in model.i)

            model.output = Constraint(model.io, model.k, rule=output_rule, doc='output constraints')

            def vrs_rule(model, io):
                return sum(model.lamda[io, i] for i in model.i) == 1

            model.vrs = Constraint(model.io, rule=vrs_rule, doc='VRS constraints')

    return model


def deaddf(y, x, gx, gy, rts):
    # rts     = "vrs": variable returns to scale
    #         = "crs": constant returns to scale

    # transform data
    x = x.tolist()
    y = y.tolist()
    
    # number of DMUs
    n = len(y)

    # number of inputs
    if type(x[0]) == int or type(x[0]) == float:
        m = 1
        x = np.asmatrix(np.array(x)).tolist()
    else:
        m = len(x[0])
        x = np.array(x).T.tolist()  

    # number of outputs
    if type(y[0]) == int or type(y[0]) == float:
        p = 1
        y = np.asmatrix(np.array(y)).tolist()
    else:
        p = len(y[0])
        y = np.array(y).T.tolist()  
            
    # directional vectors
    gx = directV.dv(gx, gy, n, m, p)[0]
    gy = directV.dv(gx, gy, n, m, p)[1]

    if m == 1:
       gx = np.asmatrix(np.array(gx)).tolist()
    else:
       gx = np.array(gx).T.tolist()
    
    if p == 1:
       gy = np.asmatrix(np.array(gy)).tolist()
    else:
       gy = np.array(gy).T.tolist()
              
    # Creation of a Concrete Model
    model = ConcreteModel()

    # Set
    model.i = Set(initialize=range(n))
    model.j = Set(initialize=range(m))
    model.k = Set(initialize=range(p))

    # Alias
    model.io = SetOf(model.i)

    # Variables
    model.theta = Var(model.io, doc='directional distance')
    model.lamda = Var(model.io, model.i, bounds=(0.0, None), doc='intensity variables')

    if rts == "crs":

        # objective
        def objective_rule(model):
            return sum(model.theta[io] for io in model.io)

        model.objective = Objective(rule=objective_rule, sense=maximize, doc='objective function')

        # Constraints
        def input_rule(model, io, j):
            arow = x[j]
            crow = gx[j]
            return sum(model.lamda[io, i] * arow[i] for i in model.i) <= arow[io] - model.theta[io] * crow[io]

        model.input = Constraint(model.io, model.j, rule=input_rule, doc='input constraints')

        def output_rule(model, io, k):
            brow = y[k]
            drow = gy[k]
            return sum(model.lamda[io, i] * brow[i] for i in model.i) >= brow[io] + model.theta[io] * drow[io]

        model.output = Constraint(model.io, model.k, rule=output_rule, doc='output constraints')

    if rts == "vrs":

        # objective
        def objective_rule(model):
            return sum(model.theta[io] for io in model.io)

        model.objective = Objective(rule=objective_rule, sense=maximize, doc='objective function')

        # Constraints
        def input_rule(model, io, j):
            arow = x[j]
            crow = gx[j]
            return sum(model.lamda[io, i] * arow[i] for i in model.i) <= arow[io] - model.theta[io] * crow[io]

        model.input = Constraint(model.io, model.j, rule=input_rule, doc='input constraints')

        def output_rule(model, io, k):
            brow = y[k]
            drow = gy[k]
            return sum(model.lamda[io, i] * brow[i] for i in model.i) >= brow[io] + model.theta[io] * drow[io]

        model.output = Constraint(model.io, model.k, rule=output_rule, doc='output constraints')

        def vrs_rule(model, io):
            return sum(model.lamda[io, i] for i in model.i) == 1

        model.vrs = Constraint(model.io, rule=vrs_rule, doc='VRS constraints')

    return model


def deaddfb(y, x, b, gx, gy, gb, rts):
    # rts     = "vrs": variable returns to scale
    #         = "crs": constant returns to scale

    # transform data
    x = x.tolist()
    y = y.tolist()
    b = b.tolist()
    
    # number of DMUs
    n = len(y)

    # number of inputs
    if type(x[0]) == int or type(x[0]) == float:
        m = 1
        x = np.asmatrix(np.array(x)).tolist()
    else:
        m = len(x[0])
        x = np.array(x).T.tolist()  

    # number of outputs
    if type(y[0]) == int or type(y[0]) == float:
        p = 1
        y = np.asmatrix(np.array(y)).tolist()
    else:
        p = len(y[0])
        y = np.array(y).T.tolist()  
        
    # number of undesirable outputs
    if type(b[0]) == int or type(b[0]) == float:
        q = 1   
        b = np.asmatrix(np.array(b)).tolist()
    else:
        q = len(b[0])
        b = np.array(b).T.tolist()
        
    # directional vectors
    gx = directV.dvb(gx, gb, gy, n, m, q, p)[0]
    gb = directV.dvb(gx, gb, gy, n, m, q, p)[1]
    gy = directV.dvb(gx, gb, gy, n, m, q, p)[2]

    if m == 1:
       gx = np.asmatrix(np.array(gx)).tolist()
    else:
       gx = np.array(gx).T.tolist()
    
    if p == 1:
       gy = np.asmatrix(np.array(gy)).tolist()
    else:
       gy = np.array(gy).T.tolist()
       
    if q == 1:
        gb = np.asmatrix(np.array(gb)).tolist()     
    else:
        gb = np.array(gb).T.tolist()

    # Creation of a Concrete Model
    model = ConcreteModel()

    # Set
    model.i = Set(initialize=range(n))
    model.j = Set(initialize=range(m))
    model.k = Set(initialize=range(p))
    model.l = Set(initialize=range(q))

    # Alias
    model.io = SetOf(model.i)

    # Variables
    model.theta = Var(model.io, doc='directional distance')
    model.lamda = Var(model.io, model.i, bounds=(0.0, None), doc='intensity variables')

    if rts == "crs":

        # objective
        def objective_rule(model):
            return sum(model.theta[io] for io in model.io)

        model.objective = Objective(rule=objective_rule, sense=maximize, doc='objective function')

        # Constraints
        def input_rule(model, io, j):
            arow = x[j]
            crow = gx[j]
            return arow[io] - model.theta[io] * crow[io] >= sum(model.lamda[io, i] * arow[i] for i in model.i)

        model.input = Constraint(model.io, model.j, rule=input_rule, doc='input constraints')

        def output_rule(model, io, k):
            brow = y[k]
            drow = gy[k]
            return brow[io] + model.theta[io] * drow[io] <= sum(model.lamda[io, i] * brow[i] for i in model.i)

        model.output = Constraint(model.io, model.k, rule=output_rule, doc='output constraints')

        def bads_rule(model, io, l):
            erow = b[l]
            frow = gb[l]
            return erow[io] - model.theta[io] * frow[io] == sum(model.lamda[io, i] * erow[i] for i in model.i)

        model.bads = Constraint(model.io, model.l, rule=bads_rule, doc='bad output constraints')

    if rts == "vrs":

        # objective
        def objective_rule(model):
            return sum(model.theta[io] for io in model.io)

        model.objective = Objective(rule=objective_rule, sense=maximize, doc='objective function')

        # Constraints
        def input_rule(model, io, j):
            arow = x[j]
            crow = gx[j]
            return arow[io] - model.theta[io] * crow[io] >= sum(model.lamda[io, i] * arow[i] for i in model.i)

        model.input = Constraint(model.io, model.j, rule=input_rule, doc='input constraints')

        def output_rule(model, io, k):
            brow = y[k]
            drow = gy[k]
            return brow[io] + model.theta[io] * drow[io] <= sum(model.lamda[io, i] * brow[i] for i in model.i)

        model.output = Constraint(model.io, model.k, rule=output_rule, doc='output constraints')

        def bads_rule(model, io, l):
            erow = b[l]
            frow = gb[l]
            return erow[io] - model.theta[io] * frow[io] == sum(model.lamda[io, i] * erow[i] for i in model.i)

        model.bads = Constraint(model.io, model.l, rule=bads_rule, doc='bad output constraints')

        def vrs_rule(model, io):
            return sum(model.lamda[io, i] for i in model.i) == 1

        model.vrs = Constraint(model.io, rule=vrs_rule, doc='VRS constraints')

    return model


def deaproj(y, x, yref, xref, orient, rts):
    # orient  = "io" : input orientation
    #         = "oo" : output orientation
    # rts     = "vrs": variable returns to scale
    #         = "crs": constant returns to scale

    # transform data
    x = x.tolist()
    y = y.tolist()
    xref = xref.tolist()
    yref = yref.tolist()

    # number of DMUs
    n1 = len(y)
    n2 = len(yref)

    # number of inputs
    if type(x[0]) == int or type(x[0]) == float:
        m = 1
        x = np.asmatrix(np.array(x)).tolist()
        xref = np.asmatrix(np.array(xref)).tolist()
    else:
        m = len(x[0])
        x = np.array(x).T.tolist()
        xref = np.array(xref).T.tolist()

    # number of outputs
    if type(y[0]) == int or type(y[0]) == float:
        p = 1
        y = np.asmatrix(np.array(y)).tolist()
        yref = np.asmatrix(np.array(yref)).tolist()
    else:
        p = len(y[0])
        y = np.array(y).T.tolist()
        yref = np.array(yref).T.tolist()

    # Creation of a Concrete Model
    model = ConcreteModel()

    # Set
    model.io = Set(initialize=range(n1))
    model.j = Set(initialize=range(m))
    model.k = Set(initialize=range(p))
    model.i = Set(initialize=range(n2))

    # Variables
    model.theta = Var(model.io, doc='efficiency')
    model.lamda = Var(model.io, model.i, bounds=(0.0, None), doc='intensity variables')
    
    if orient == "io":

        if rts == "crs":
            
            # objective
            def objective_rule(model):
                return sum(model.theta[io] for io in model.io)

            model.objective = Objective(rule=objective_rule, sense=minimize, doc='objective function')

            # Constraints
            def input_rule(model, io, j):
                arow = x[j]
                crow = xref[j]
                return (model.theta[io] * arow[io]) >= sum(model.lamda[io, i] * crow[i] for i in model.i)

            model.input = Constraint(model.io, model.j, rule=input_rule, doc='input constraints')

            def output_rule(model, io, k):
                brow = y[k]
                drow = yref[k]
                return sum(model.lamda[io, i] * drow[i] for i in model.i) >= brow[io]

            model.output = Constraint(model.io, model.k, rule=output_rule, doc='output constraints')

        if rts == "vrs":
            
            # objective
            def objective_rule(model):
                return sum(model.theta[io] for io in model.io)

            model.objective = Objective(rule=objective_rule, sense=minimize, doc='objective function')

            # Constraints
            def input_rule(model, io, j):
                arow = x[j]
                crow = xref[j]
                return (model.theta[io] * arow[io]) >= sum(model.lamda[io, i] * crow[i] for i in model.i)

            model.input = Constraint(model.io, model.j, rule=input_rule, doc='input constraints')

            def output_rule(model, io, k):
                brow = y[k]
                drow = yref[k]
                return sum(model.lamda[io, i] * drow[i] for i in model.i) >= brow[io]

            model.output = Constraint(model.io, model.k, rule=output_rule, doc='output constraints')

            def vrs_rule(model, io):
                return sum(model.lamda[io, i] for i in model.i) == 1

            model.vrs = Constraint(model.io, rule=vrs_rule, doc='VRS constraints')

    if orient == "oo":

        if rts == "crs":

            # objective
            def objective_rule(model):
                return sum(model.theta[io] for io in model.io)

            model.objective = Objective(rule=objective_rule, sense=maximize, doc='objective function')

            # Constraints
            def input_rule(model, io, j):
                arow = x[j]
                crow = xref[j]
                return sum(model.lamda[io, i] * crow[i] for i in model.i) <= arow[io]

            model.input = Constraint(model.io, model.j, rule=input_rule, doc='input constraints')

            def output_rule(model, io, k):
                brow = y[k]
                drow = yref[k]
                return model.theta[io] * brow[io] <= sum(model.lamda[io, i] * drow[i] for i in model.i)

            model.output = Constraint(model.io, model.k, rule=output_rule, doc='output constraints')

        if rts == "vrs":

            # objective
            def objective_rule(model):
                return sum(model.theta[io] for io in model.io)

            model.objective = Objective(rule=objective_rule, sense=maximize, doc='objective function')

            # Constraints
            def input_rule(model, io, j):
                arow = x[j]
                crow = xref[j]
                return sum(model.lamda[io, i] * crow[i] for i in model.i) <= arow[io]

            model.input = Constraint(model.io, model.j, rule=input_rule, doc='input constraints')

            def output_rule(model, io, k):
                brow = y[k]
                drow = yref[k]
                return model.theta[io] * brow[io] <= sum(model.lamda[io, i] * drow[i] for i in model.i)

            model.output = Constraint(model.io, model.k, rule=output_rule, doc='output constraints')

            def vrs_rule(model, io):
                return sum(model.lamda[io, i] for i in model.i) == 1

            model.vrs = Constraint(model.io, rule=vrs_rule, doc='VRS constraints')        

    return model


def deaddfproj(y, x, yref, xref, gx, gy, rts):
    # rts     = "vrs": variable returns to scale
    #         = "crs": constant returns to scale

    # transform data
    x = x.tolist()
    y = y.tolist()
    xref = xref.tolist()
    yref = yref.tolist()

    # number of DMUs
    n1 = len(y)
    n2 = len(yref)
    
    # number of inputs
    if type(x[0]) == int or type(x[0]) == float:
        m = 1
        x = np.asmatrix(np.array(x)).tolist()
        xref = np.asmatrix(np.array(xref)).tolist()
    else:
        m = len(x[0])
        x = np.array(x).T.tolist()
        xref = np.array(xref).T.tolist() 

    # number of outputs
    if type(y[0]) == int or type(y[0]) == float:
        p = 1
        y = np.asmatrix(np.array(y)).tolist()
        yref = np.asmatrix(np.array(yref)).tolist()
    else:
        p = len(y[0])
        y = np.array(y).T.tolist()  
        yref = np.array(yref).T.tolist() 
            
    # directional vectors
    gx = directV.dv(gx, gy, n1, m, p)[0]
    gy = directV.dv(gx, gy, n1, m, p)[1]

    if m == 1:
       gx = np.asmatrix(np.array(gx)).tolist()
    else:
       gx = np.array(gx).T.tolist()
    
    if p == 1:
       gy = np.asmatrix(np.array(gy)).tolist()
    else:
       gy = np.array(gy).T.tolist()
              
    # Creation of a Concrete Model
    model = ConcreteModel()

    # Set
    model.io = Set(initialize=range(n1))
    model.j = Set(initialize=range(m))
    model.k = Set(initialize=range(p))
    model.i = Set(initialize=range(n2))

    # Variables
    model.theta = Var(model.io, doc='directional distance')
    model.lamda = Var(model.io, model.i, bounds=(0.0, None), doc='intensity variables')

    if rts == "crs":

        # objective
        def objective_rule(model):
            return sum(model.theta[io] for io in model.io)

        model.objective = Objective(rule=objective_rule, sense=maximize, doc='objective function')

        # Constraints
        def input_rule(model, io, j):
            arow = x[j]
            crow = gx[j]
            erow = xref[j]
            return sum(model.lamda[io, i] * erow[i] for i in model.i) <= arow[io] - model.theta[io] * crow[io]

        model.input = Constraint(model.io, model.j, rule=input_rule, doc='input constraints')

        def output_rule(model, io, k):
            brow = y[k]
            drow = gy[k]
            frow = yref[k]
            return sum(model.lamda[io, i] * frow[i] for i in model.i) >= brow[io] + model.theta[io] * drow[io]

        model.output = Constraint(model.io, model.k, rule=output_rule, doc='output constraints')

    if rts == "vrs":

        # objective
        def objective_rule(model):
            return sum(model.theta[io] for io in model.io)

        model.objective = Objective(rule=objective_rule, sense=maximize, doc='objective function')

        # Constraints
        def input_rule(model, io, j):
            arow = x[j]
            crow = gx[j]
            erow = xref[j]
            return sum(model.lamda[io, i] * erow[i] for i in model.i) <= arow[io] - model.theta[io] * crow[io]

        model.input = Constraint(model.io, model.j, rule=input_rule, doc='input constraints')

        def output_rule(model, io, k):
            brow = y[k]
            drow = gy[k]
            frow = yref[k]
            return sum(model.lamda[io, i] * frow[i] for i in model.i) >= brow[io] + model.theta[io] * drow[io]

        model.output = Constraint(model.io, model.k, rule=output_rule, doc='output constraints')

        def vrs_rule(model, io):
            return sum(model.lamda[io, i] for i in model.i) == 1

        model.vrs = Constraint(model.io, rule=vrs_rule, doc='VRS constraints')
    
    return model


def deaddfbproj(y, x, b, yref, xref, bref, gx, gy, gb, rts):
    # rts     = "vrs": variable returns to scale
    #         = "crs": constant returns to scale

    # transform data
    x = x.tolist()
    y = y.tolist()
    b = b.tolist()
    xref = xref.tolist()
    yref = yref.tolist()
    bref = bref.tolist()

    # number of DMUs
    n1 = len(y)
    n2 = len(yref)

    # number of inputs
    if type(x[0]) == int or type(x[0]) == float:
        m = 1
        x = np.asmatrix(np.array(x)).tolist()
        xref = np.asmatrix(np.array(xref)).tolist()
    else:
        m = len(x[0])
        x = np.array(x).T.tolist()  
        xref = np.array(xref).T.tolist()  

    # number of outputs
    if type(y[0]) == int or type(y[0]) == float:
        p = 1
        y = np.asmatrix(np.array(y)).tolist()
        yref = np.asmatrix(np.array(yref)).tolist()
    else:
        p = len(y[0])
        y = np.array(y).T.tolist()  
        yref = np.array(yref).T.tolist()  
        
    # number of undesirable outputs
    if type(b[0]) == int or type(b[0]) == float:
        q = 1   
        b = np.asmatrix(np.array(b)).tolist()
        bref = np.asmatrix(np.array(bref)).tolist()
    else:
        q = len(b[0])
        b = np.array(b).T.tolist()
        bref = np.array(bref).T.tolist()
        
    # directional vectors
    gx = directV.dvb(gx, gb, gy, n1, m, q, p)[0]
    gb = directV.dvb(gx, gb, gy, n1, m, q, p)[1]
    gy = directV.dvb(gx, gb, gy, n1, m, q, p)[2]

    if m == 1:
       gx = np.asmatrix(np.array(gx)).tolist()
    else:
       gx = np.array(gx).T.tolist()
    
    if p == 1:
       gy = np.asmatrix(np.array(gy)).tolist()
    else:
       gy = np.array(gy).T.tolist()
       
    if q == 1:
        gb = np.asmatrix(np.array(gb)).tolist()     
    else:
        gb = np.array(gb).T.tolist()

    # Creation of a Concrete Model
    model = ConcreteModel()

    # Set
    model.io = Set(initialize=range(n1))
    model.j = Set(initialize=range(m))
    model.k = Set(initialize=range(p))
    model.l = Set(initialize=range(q))
    model.i = Set(initialize=range(n2))

    # Variables
    model.theta = Var(model.io, doc='directional distance')
    model.lamda = Var(model.io, model.i, bounds=(0.0, None), doc='intensity variables')

    if rts == "crs":

        # objective
        def objective_rule(model):
            return sum(model.theta[io] for io in model.io)

        model.objective = Objective(rule=objective_rule, sense=maximize, doc='objective function')

        # Constraints
        def input_rule(model, io, j):
            arow = x[j]
            crow = gx[j]
            erow = xref[j]
            return arow[io] - model.theta[io] * crow[io] >= sum(model.lamda[io, i] * erow[i] for i in model.i)

        model.input = Constraint(model.io, model.j, rule=input_rule, doc='input constraints')

        def output_rule(model, io, k):
            brow = y[k]
            drow = gy[k]
            frow = yref[k]
            return brow[io] + model.theta[io] * drow[io] <= sum(model.lamda[io, i] * frow[i] for i in model.i)

        model.output = Constraint(model.io, model.k, rule=output_rule, doc='output constraints')

        def bads_rule(model, io, l):
            erow = b[l]
            frow = gb[l]
            grow = bref[l]
            return erow[io] - model.theta[io] * frow[io] == sum(model.lamda[io, i] * grow[i] for i in model.i)

        model.bads = Constraint(model.io, model.l, rule=bads_rule, doc='bad output constraints')

    if rts == "vrs":

        # objective
        def objective_rule(model):
            return sum(model.theta[io] for io in model.io)

        model.objective = Objective(rule=objective_rule, sense=maximize, doc='objective function')

        # Constraints
        def input_rule(model, io, j):
            arow = x[j]
            crow = gx[j]
            erow = xref[j]
            return arow[io] - model.theta[io] * crow[io] >= sum(model.lamda[io, i] * erow[i] for i in model.i)

        model.input = Constraint(model.io, model.j, rule=input_rule, doc='input constraints')

        def output_rule(model, io, k):
            brow = y[k]
            drow = gy[k]
            frow = yref[k]
            return brow[io] + model.theta[io] * drow[io] <= sum(model.lamda[io, i] * frow[i] for i in model.i)

        model.output = Constraint(model.io, model.k, rule=output_rule, doc='output constraints')

        def bads_rule(model, io, l):
            erow = b[l]
            frow = gb[l]
            grow = bref[l]
            return erow[io] - model.theta[io] * frow[io] == sum(model.lamda[io, i] * grow[i] for i in model.i)

        model.bads = Constraint(model.io, model.l, rule=bads_rule, doc='bad output constraints')

        def vrs_rule(model, io):
            return sum(model.lamda[io, i] for i in model.i) == 1

        model.vrs = Constraint(model.io, rule=vrs_rule, doc='VRS constraints')

    return model