import numpy as np
from . import CNLSG1, CNLSG2, sweet


def CNLSG(y, x, cet, fun, rts):
    # cet     = "addi" : Additive composite error term
    #         = "mult" : Multiplicative composite error term
    # fun     = "prod" : production frontier
    #         = "cost" : cost frontier
    # rts     = "vrs"  : variable returns to scale
    #         = "crs"  : constant returns to scale

    Cutactive = sweet.sweet(x)
        
    # solve the initial CNLS+G model to obtain initial solution
    model1 = CNLSG1.CNLSG1(y, x, Cutactive, cet, fun, rts)
    model1.optimize(remote=False)

    alpha1 = model1.get_alpha1()
    beta1 = model1.get_beta1()

    x = np.asarray(x)
    # active (added) violated concavity constraint by iterative procedure
    Active = np.zeros((len(x), len(x)))
    # violated concavity constraint
    Active2 = np.zeros((len(x), len(x)))

    # go into the loop
    Activetmp1 = 0.0
    for i in range(len(x)):
        Activetmp = 0.0
        # go into the sub-loop and find the violated concavity constraints   
        for j in range(len(x)):
            if cet == "addi":            
                if rts == "vrs":
                    if fun == "prod":
                         Active2[i, j] = alpha1[i] + np.sum(beta1[i, :] * x[i, :]) - \
                                         alpha1[j] - np.sum(beta1[j, :] * x[i, :])
                    elif fun == "cost":
                         Active2[i, j] = - alpha1[i] - np.sum(beta1[i, :] * x[i, :]) + \
                                           alpha1[j] + np.sum(beta1[j, :] * x[i, :])
            if cet == "mult":
                if rts == "vrs":
                    if fun == "prod":
                         Active2[i, j] = alpha1[i] + np.sum(beta1[i, :] * x[i, :]) - \
                                         alpha1[j] - np.sum(beta1[j, :] * x[i, :])
                    elif fun == "cost":
                         Active2[i, j] = - alpha1[i] - np.sum(beta1[i, :] * x[i, :]) + \
                                           alpha1[j] + np.sum(beta1[j, :] * x[i, :])
                if rts == "crs":  
                    if fun == "prod":
                         Active2[i, j] = np.sum(beta1[i, :] * x[i, :]) - \
                                         np.sum(beta1[j, :] * x[i, :])
                    elif fun == "cost":
                         Active2[i, j] = - np.sum(beta1[i, :] * x[i, :]) + \
                                           np.sum(beta1[j, :] * x[i, :])    
            if Active2[i, j] > Activetmp:
                Activetmp = Active2[i, j]
        for j in range(len(x)):        
            if Active2[i,j] >= Activetmp and Activetmp >0:
                Active[i, j] = 1
        # find the maximal violated constraint in sub-loop and added into the active matrix
        if Activetmp > Activetmp1:
              Activetmp1 = Activetmp

    # solve the CNLS+G model iteratively
    """the stopping criteria of algorithm i.e. there is no violated constraint, 0.0001 is for rounding error."""

    while Activetmp > 0.0001:

        # solve the CNLS+G iteratively
        model2 = CNLSG2.CNLSG2(y, x, Cutactive, Active, cet, fun, rts)
        model2.optimize(remote=False)

        alpha2 = model2.get_alpha2()
        beta2 = model2.get_beta2()

        Activetmp1 = 0.0
        # go into the loop
        for i in range(len(x)):
            Activetmp = 0.0
            # go into the sub-loop and find the violated concavity constraints
            for j in range(len(x)):
                if cet == "addi":
                    if rts == "vrs":
                        if fun == "prod":
                            Active2[i, j] = alpha2[i] + np.sum(beta2[i, :] * x[i, :]) - \
                                            alpha2[j] - np.sum(beta2[j, :] * x[i, :])
                        elif fun == "cost":
                            Active2[i, j] = - alpha2[i] - np.sum(beta2[i, :] * x[i, :]) + \
                                              alpha2[j] + np.sum(beta2[j, :] * x[i, :])
                if cet == "mult":
                    if rts == "vrs":
                        if fun == "prod":
                            Active2[i, j] = alpha2[i] + np.sum(beta2[i, :] * x[i, :]) - \
                                            alpha2[j] - np.sum(beta2[j, :] * x[i, :])
                        elif fun == "cost":
                            Active2[i, j] = - alpha2[i] - np.sum(beta2[i, :] * x[i, :]) + \
                                              alpha2[j] + np.sum(beta2[j, :] * x[i, :])
                    if rts == "crs":
                        if fun == "prod":
                            Active2[i, j] = np.sum(beta2[i, :] * x[i, :]) - \
                                            np.sum(beta2[j, :] * x[i, :])
                        elif fun == "cost":
                            Active2[i, j] = - np.sum(beta2[i, :] * x[i, :]) + \
                                              np.sum(beta2[j, :] * x[i, :])
                if Active2[i, j] > Activetmp:
                    Activetmp = Active2[i, j]
            # find the maximal violated constraint in sub-loop and added into the active matrix
            for j in range(len(x)):
                if Active2[i, j] >= Activetmp and Activetmp > 0:
                    Active[i, j] = 1
            if Activetmp > Activetmp1:
                Activetmp1 = Activetmp
       
    epsilon2 = model2.get_residual2()
    
    return alpha2, beta2, epsilon2
