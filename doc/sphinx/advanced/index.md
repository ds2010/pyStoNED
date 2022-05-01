# Advanced topics

## Removing the constraint on Beta

We can remove the constrinat on Beta by adding `model.__model__.beta.setlb(None)` before the model optimization (i.e., `model.optimize()`). Now the estimated Beta can take any value in Reals. For example, the constraint Beta >= 0 is removed in CQR:

```python
# import packages
from pystoned import CQER
from pystoned.constant import CET_ADDI, FUN_PROD, OPT_LOCAL, RTS_VRS
from pystoned import dataset as dataset

# import the GHG example data
data = dataset.load_GHG_abatement_cost(x_select=['HRSN', 'CPNK', 'GHG'], y_select=['VALK'])

# calculate the quantile model
model = CQER.CQR(y=data.y, x=data.x, tau=0.5, z=None, cet=CET_ADDI, fun=FUN_PROD, rts=RTS_VRS)
# remove the constraint beta >= 0
model.__model__.beta.setlb(None)
model.optimize(OPT_LOCAL)

# display estimated beta
model.display_beta()
```

## Adding an additional constraint 

The extral constraint, e.g., 0 <= Beta <= 1, can be added using

```python
from pyomo.environ import Constraint

def constraint_rule(model, i, j):
    upperbound = [1.0, 1.0, 1.0]
    return model.beta[i, j] <= upperbound[j]

res.__model__.beta_constraint_rule = Constraint(res.__model__.I,
                                                res.__model__.J,
                                                rule=constraint_rule,
                                                doc='beta constraint')
```

A simple CQR example:

```python
# import packages
from pystoned import CQER
from pystoned.constant import CET_ADDI, FUN_PROD, OPT_LOCAL, RTS_VRS
from pystoned import dataset as dataset
from pyomo.environ import Constraint

# import the GHG example data
data = dataset.load_GHG_abatement_cost(x_select=['HRSN', 'CPNK', 'GHG'], y_select=['VALK'])

# calculate the quantile model
res = CQER.CQR(y=data.y, x=data.x, tau=0.5, z=None, cet=CET_ADDI, fun=FUN_PROD, rts=RTS_VRS)
# add an extral constraint on beta
def constraint_rule(model, i, j):
    upperbound = [1.0, 1.0, 1.0]
    return model.beta[i, j] <= upperbound[j]

res.__model__.beta_constraint_rule = Constraint(res.__model__.I,
                                                res.__model__.J,
                                                rule=constraint_rule,
                                                doc='beta constraint')
res.optimize(OPT_LOCAL)

# display estimated beta
res.display_beta()
```

## Miscellaneous

### gams2python

[pyomo](http://www.pyomo.org/) provides a good coding environment that can help us smoothly transfer
from GAMS to Python. Thus, we prepare a short tutorial to help GAMSers to understand how to rewrite 
the CNLS models, even other complicated models on Python.

[A short comparison](https://colab.research.google.com/github/ds2010/pyStoNED/blob/master/notebooks/gams2python.ipynb):

```python
### We first present the GAMS code,
#sets 
#     i        "DMU's"  /1*10/
#     j        'outputs' /Energy, Length, Customers/;
    
### then the Python code is added to compare.
from pyomo.environ import *
model = ConcreteModel(name = "CNLS")
model.i = Set(initialize=['i1', 'i2', 'i3', 'i4', 'i5', 'i6','i7', 'i8', 'i9','i10'], doc='DMUS', ordered=True)
model.j = Set(initialize=['Energy', 'Length', 'Customers'], doc='outputs')

# alias(i,h); 

model.h = SetOf(model.i)

#Table data(i,j)
#$Include energy.txt;

dtab = {
        ('i1',  'Energy')   : 75,
        ('i1',  'Length')   : 878,
        ('i1',  'Customers'): 4933,
        ('i2',  'Energy')   : 62,
        ('i2',  'Length')   : 964,
        ('i2',  'Customers'): 6149,
        ('i3',  'Energy')   : 78,
        ('i3',  'Length')   : 676,
        ('i3',  'Customers'): 6098,
        ('i4',  'Energy')   : 683,
        ('i4',  'Length')   : 12522,
        ('i4',  'Customers'): 55226,
        ('i5',  'Energy')   : 27,
        ('i5',  'Length')   : 697,
        ('i5',  'Customers'): 1670,
        ('i6',  'Energy')   : 295,
        ('i6',  'Length')   : 953,
        ('i6',  'Customers'): 22949,
        ('i7',  'Energy')   : 44,
        ('i7',  'Length')   : 917,
        ('i7',  'Customers'): 3599,
        ('i8',  'Energy')   : 171,
        ('i8',  'Length')   : 1580,
        ('i8',  'Customers'): 11081,
        ('i9',  'Energy')   : 98,
        ('i9',  'Length')   : 116,
        ('i9',  'Customers'): 377,
        ('i10', 'Energy')   : 203,
        ('i10', 'Length')   : 740,
        ('i10', 'Customers'): 10134,
        }
model.d = Param(model.i, model.j, initialize=dtab, doc='output data')

#PARAMETERS
#c(i)         "Total cost of firm i"
#y(i,j)       "Output j of firm i";
model.c = Param(model.i, initialize={'i1': 1612,
                                     'i2': 1659,
                                     'i3': 1708,
                                     'i4': 18918,
                                     'i5': 1167,
                                     'i6': 3395,
                                     'i7': 1333,
                                     'i8': 3518,
                                     'i9': 1415,
                                     'i10':2469,
                                    }, 
                      doc='Cost data')

def y_init(model, i, j):
    return  model.d[i, j]
model.y = Param(model.i, model.j, initialize=y_init, doc='output data')

#VARIABLES
#E(i)            "Composite error term (v + u)"
#SSE             "Sum of squares of residuals";
#POSITIVE VARIABLES
#b(i,j)    "Beta-coefficients (positivity ensures monotonicity)"
#Chat(i)  ;

model.b = Var(model.i, model.j, bounds=(0.0,None), doc='beta-coeff')
model.e = Var(model.i, doc='res')
model.f = Var(model.i, bounds=(0.0,None), doc='frontier')

#Equations
#QSSE                  objective function = sum of squares of residuals
#QREGRESSION(i)        log-transformed regression equation
#Qlog(i)               supporting hyperplanes of the nonparametric cost function
#QCONC(i,h)            concavity constraint (Afriat inequalities);

#QSSE..                SSE=e=sum(i,E(i)*E(i)) ;
#QREGRESSION(i)..      log(C(i)) =e= log(Chat(i) + 1) + E(i);
#Qlog(i)..             Chat(i) =e= sum(j, b(i,j)*Y(i,j)) - 1;
#QCONC(i,h)..          sum(j, b(i,j)*Y(i,j)) =g= sum(j, b(h,j)*Y(i,j));

def objective_rule(model):
    return sum(model.e[i]*model.e[i] for i in model.i)
model.objective = Objective(rule=objective_rule, sense=minimize, doc='objective function')

def qreg_rule(model, i):
    return log(model.c[i]) == log(model.f[i] + 1) + model.e[i]
model.qreg = Constraint(model.i, rule=qreg_rule, doc='log-transformed regression')

def qlog_rule(model, i):
    return model.f[i] == sum(model.b[i, j]*model.y[i, j] for j in model.j) - 1
model.qlog = Constraint(model.i, rule=qlog_rule, doc='cost function')

def qconvex_rule(model, i, h):
    return sum(model.b[i,j]*model.y[i,j] for j in model.j) >= sum(model.b[h,j]*model.y[i,j] for j in model.j)
model.qconvex = Constraint(model.i, model.h, rule=qconvex_rule, doc='convexity constraint')

# Execute the model
#MODEL StoNED /all/;
#SOLVE StoNED using NLP Minimizing SSE;
    
from pyomo.opt import SolverFactory
import pyomo.environ
from os import environ
solver_manager = SolverManagerFactory('neos')
environ['NEOS_EMAIL'] = 'email@address'
results = solver_manager.solve(model, opt='minos')

#display E.l, b.l;

model.e.display()
model.b.display()      
```

### CNLS_ConcreteModel

We also prepare a concrete model that does not need to call any of our developed functions in the pyStoNED package.
In this concrete model, one can define the parameter specification by reading data from an Excel file.

[A concreteModel for CNLS estimation](https://colab.research.google.com/github/ds2010/pyStoNED/blob/master/notebooks/CNLS_ConcreteModel.ipynb):

```python
# Import packages
import pd as pd
import numpy as np
import sys
from pyomo.environ import *

# Sets
i = np.array(['i{}'.format(i) for i in range(1, 90)])
model.i = Set(initialize=i, doc='DMUs', ordered=True )
model.j = Set(initialize=['j1', 'j2', 'j3'], doc='inputs and outputs')

# alias
model.h = SetOf(model.i) 

# Parameters 
# output (y1,y2,y3), avaiable at https://raw.githubusercontent.com/ds2010/pyStoNED/master/pystoned/data/electricityFirms.csv.
df1 = pd.read_excel('y.xlsx', header=0, index_col=0)
Dict1 = dict()
for i in df1.index:
    for j in df1.columns:
        Dict1[i, j] = float(df1[j][i])
model.y = Param(model.i, model.j, initialize = Dict1) 

# input (cost)
df2 = pd.read_excel('x.xlsx', header=0, index_col=0)
Dict2 = dict()
for i in df2.index:
    Dict2[i] = float(df2['TOTEX'][i])
model.c = Param(model.i, initialize = Dict2) 

#Variables
model.b = Var(model.i, model.j, bounds=(0.0,None), doc='beta-coeff')
model.e = Var(model.i, doc='res')
model.f = Var(model.i, bounds=(0.0,None), doc='frontier')

# Constraints and objective f.
def qreg_rule(model, i):
    return log(model.c[i]) == log(model.f[i] + 1) + model.e[i]
model.qreg = Constraint(model.i, rule=qreg_rule, doc='log-transformed regression')

def qlog_rule(model, i):
    return model.f[i] == sum(model.b[i, j]*model.y[i, j] for j in model.j) - 1
model.qlog = Constraint(model.i, rule=qlog_rule, doc='cost function')

def qconcav_rule(model, i, h):
    return sum(model.b[i,j]*model.y[i,j] for j in model.j) >= sum(model.b[h,j]*model.y[i,j] for j in model.j)
model.qconcav = Constraint(model.i, model.h, rule=qconcav_rule, doc='concavity constraint')

def objective_rule(model):
    return sum(model.e[i]*model.e[i] for i in model.i)
model.objective = Objective(rule=objective_rule, sense=minimize, doc='Define objective function')

#Solve the model
from os import environ
solver_manager = SolverManagerFactory('neos')
environ['NEOS_EMAIL'] = 'email@address'
results = solver_manager.solve(model, opt='knitro', tee=True)

#retrive the beta
ind = list(model.b)
val = list(model.b[:,:].value)
df_b = [ i + tuple([j]) for i, j in zip(ind, val)]
df_b = np.asarray(df_b)
# display beta coefficients
df_b

df_b = pd.DataFrame(df_b,columns = ['Name', 'Key', 'Value'])
df_b = df_b.pivot(index='Name', columns='Key', values='Value')
# retrive the residuals
ind = list(model.e)
val = list(model.e[:].value)
df_e = np.asarray(val)
#display residuals
df_e
```


### DEA_ConcreteModel

We also prepare a concrete model that 
can be used to calculate the input oriented VRS model.

 [A ConcreteModel for DEA estimation](https://colab.research.google.com/github/ds2010/pyStoNED/blob/master/notebooks/DEA_ConcreteModel.ipynb):

```python
# import PYOMO package
from pyomo.environ import *
# creat a concrete model
model = ConcreteModel()

# Sets
model.i = Set(initialize=['i1', 'i2','i3', 'i4','i5', 'i6','i7', 'i8','i9', 'i10'], doc='DMUs', ordered=True)
model.j = Set(initialize=['OPEX', 'CAPEX'])   # inputs
model.k = Set(initialize=['Energy', 'Length', 'Customers'])   #outputs

# Alias
model.io = SetOf(model.i) 

# Parameters (define and import the input-output data)
inputdata = {
                ('OPEX', 'i1') :   681,
                ('OPEX', 'i2') :   559,
                ('OPEX', 'i3') :   836,
                ('OPEX', 'i4') :   7559,
                ('OPEX', 'i5') :   424,
                ('OPEX', 'i6') :   1483,
                ('OPEX', 'i7') :   658,
                ('OPEX', 'i8') :   1433,
                ('OPEX', 'i9') :   850,
                ('OPEX', 'i10') :  1155,
                ('CAPEX', 'i1') :  729,
                ('CAPEX', 'i2') :  673,
                ('CAPEX', 'i3') :  851,
                ('CAPEX', 'i4') :  8384,
                ('CAPEX', 'i5') :  562,
                ('CAPEX', 'i6') :  1587,
                ('CAPEX', 'i7') :  570,
                ('CAPEX', 'i8') :  1311,
                ('CAPEX', 'i9') :  564,
                ('CAPEX', 'i10'):  1108,
                }
model.x = Param(model.j, model.i, initialize=inputdata)

outputdata = {
                ('Energy', 'i1'):   75,
                ('Energy', 'i2'):   62,
                ('Energy', 'i3'):   78,
                ('Energy', 'i4'):   683,
                ('Energy', 'i5'):   27,
                ('Energy', 'i6'):   295,
                ('Energy', 'i7'):   44,
                ('Energy', 'i8'):   171,
                ('Energy', 'i9'):   98,
                ('Energy', 'i10'):  203,
                ('Length', 'i1'):   878,
                ('Length', 'i2'):   964,
                ('Length', 'i3'):   676,
                ('Length', 'i4'):   12522,
                ('Length', 'i5'):   697,
                ('Length', 'i6'):   953,
                ('Length', 'i7'):   917,
                ('Length', 'i8'):   1580,
                ('Length', 'i9'):   116,
                ('Length', 'i10'):  740,
                ('Customers', 'i1'):4933,
                ('Customers', 'i2'):6149,
                ('Customers', 'i3'):6098,
                ('Customers', 'i4'):55226,
                ('Customers', 'i5'):1670,
                ('Customers', 'i6'):22949,
                ('Customers', 'i7'):3599,
                ('Customers', 'i8'):11081,
                ('Customers', 'i9'):377,
                ('Customers', 'i10'):10134,
                }
model.y = Param(model.k, model.i, initialize=outputdata)  

# Variables
model.lamda = Var(model.io, model.i, bounds=(0.0, None), doc='envelopment efficiency') 
model.theta = Var(model.io, doc='intensity variable') 

# objective
def objective_rule(model):
    return sum(model.theta[io] for io in model.io)
model.objective = Objective(rule=objective_rule, sense=minimize, doc='Define objective function')       
# Constraints
def output_rule(model, io, k):
    return sum(model.lamda[io, i] * model.y[k, i] for i in model.i) >= model.y[k, io]
model.output = Constraint(model.io, model.k, rule=output_rule, doc='output constraints')
def input_rule(model, io, j):
    return (model.theta[io] * model.x[j, io]) >= sum(
             model.lamda[io, i] * model.x[j, i] for i in model.i)
model.input = Constraint(model.io, model.j, rule=input_rule, doc='input constraints')
def vrs_rule(model, io):
    return sum(model.lamda[io, i] for i in model.i) == 1
model.vrs = Constraint(model.io, rule=vrs_rule, doc='VRS constraints')

# calculate the DEA model 
from pyomo.opt import SolverFactory
from os import environ
solver_manager = SolverManagerFactory('neos')
environ['NEOS_EMAIL'] = 'email@address'
results = solver_manager.solve(model, opt='cplex')

# display the estimates
# efficiency
model.theta.display()
# intensity
model.lamda.display()
```