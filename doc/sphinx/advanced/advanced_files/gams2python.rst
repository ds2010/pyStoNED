=============================
Rewrite GAMS codes in Python
=============================

`PYOMO <http://www.pyomo.org/>`_ provides a good coding environment that can help us smoothly transfer
from GAMS to Python. Thus, we prepare a short tutorial to help GAMSers to understand how to rewrite 
the CNLS models, even other complicated models on Python.

A brief comparison `[.ipynb] <https://colab.research.google.com/github/ds2010/pyStoNED/blob/master/notebooks/gams2python.ipynb>`_ :

.. code:: python
    
    ### We first present the GAMS code,
    #sets 
    #     i        "DMU's"  /1*10/
    #     j        'outputs' /Energy, Length, Customers/;
    
    ### then the Python code is added to compare.
    from pyomo.environ import *
    model = ConcreteModel(name = "CNLS")
    model.i = Set(initialize=['i1', 'i2', 'i3', 'i4', 'i5', 'i6','i7', 'i8', 'i9','i10'], doc='DMUS', ordered=True)
    model.j = Set(initialize=['Energy', 'Length', 'Customers'], doc='outputs')

::

    # alias(i,h); 

    model.h = SetOf(model.i)

::

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

::

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

::

    #VARIABLES
    #E(i)            "Composite error term (v + u)"
    #SSE             "Sum of squares of residuals";
    #POSITIVE VARIABLES
    #b(i,j)    "Beta-coefficients (positivity ensures monotonicity)"
    #Chat(i)  ;

    model.b = Var(model.i, model.j, bounds=(0.0,None), doc='beta-coeff')
    model.e = Var(model.i, doc='res')
    model.f = Var(model.i, bounds=(0.0,None), doc='frontier')

::

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

::

    # Execute the model
    #MODEL StoNED /all/;
    #SOLVE StoNED using NLP Minimizing SSE;
    
    from pyomo.opt import SolverFactory
    import pyomo.environ
    solver_manager = SolverManagerFactory('neos')
    results = solver_manager.solve(model, opt='minos')

::

    #display E.l, b.l;

    model.e.display()
    model.b.display()  