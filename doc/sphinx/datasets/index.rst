.. _datasets:

Datasets
===========

In this section, the package provides four example datasets: First two are used in large number of CNLS/StoNED
liturature; the others are commonly used in the SFA liturature. In the Examples, our tutorials
will resort to these example data.

.. toctree::
    :maxdepth: 2

    firms/index.md
    countries/index.md
    front41Data/index.md
    riceProdPhil/index.md

Import internal data
-----------------------------------------

- Finnish electricity firm data

.. code:: python

    # import dataset module
    from pystoned.dataset import load_Finnish_electricity_firm

    # import all data (including the contextual varibale)
    data = load_Finnish_electricity_firm(x_select=['Energy', 'Length', 'Customers'], 
                                            y_select=['TOTEX'], 
                                            z_select=['PerUndGr'])
    x, y, z = data.x, data.y, data.z
    
    # print data
    print(x)
    print(y)
    print(z)

    # (OR) import data (only inputs and output)
    data = load_Finnish_electricity_firm(x_select=['Energy', 'Length', 'Customers'], 
                                            y_select=['TOTEX'])
    x, y = data.x, data.y
    
    # print data
    print(x)
    print(y)


- import OECD GHG emissions data

.. code:: python

    # import dataset module
    from pystoned.dataset import load_GHG_abatement_cost

    # import all data 
    data = load_GHG_abatement_cost(x_select=['HRSN', 'CPNK'], 
                                    y_select=['VALK'], 
                                    b_select=['GHG'])
    x, y, b = data.x, data.y, data.b

    # print data
    print(x)
    print(y)
    print(b)

- import Tim Coelli’s Frontier 4.1 data

.. code:: python

    # import dataset module
    from pystoned.dataset import load_Tim_Coelli_frontier

    # import all data 
    data = load_Tim_Coelli_frontier(x_select=['capital', 'labour'], 
                                        y_select=['output'])
    x, y = data.x, data.y

    # print data
    print(x)
    print(y)

- import rice production data

.. code:: python

    # import dataset module
    from pystoned.dataset import load_Philipines_rice_production

    # import all data 
    data = load_Philipines_rice_production(x_select=['AREA', 'LABOR', 'NPK', 'OTHER', 'AREAP', 'LABORP', 'NPKP', 'OTHERP'], 
                                                y_select=['PROD', 'PRICE'])
    x, y = data.x, data.y

    # print data
    print(x)
    print(y)

    # (OR) import partial data (two input-one output) 
    data = load_Philipines_rice_production(x_select=['LABOR', 'NPK'], 
                                                y_select=['PROD'])
    x, y = data.x, data.y

    # print data
    print(x)
    print(y)


Import external data
--------------------------------

Assuming that we have a dataset like the following example in `Book1.xlsx`, we then 
use the Panda to read the Excel file and organize the data using the Numpy.

+------------+------------+-----------+------------+------------+------------+
| ID         | output     | input1    | input2     | input3     | z_var      |
+============+============+===========+============+============+============+
| i1         | 120        | 10        | 55         | 103        | 0.8        |
+------------+------------+-----------+------------+------------+------------+
| i2         | 80         | 30        | 49         | 120        | 0.6        |
+------------+------------+-----------+------------+------------+------------+
| i3         | 90         | 25        | 72         | 150        | 0.3        |
+------------+------------+-----------+------------+------------+------------+
| i4         | 110        | 16        | 39         | 100        | 0.5        |
+------------+------------+-----------+------------+------------+------------+
| ...        | ...        | ...       | ...        | ...        | ...        |
+------------+------------+-----------+------------+------------+------------+

.. code:: python

    # import basic modules
    import numpy as np
    import pandas as pd

    # import Excel data 
    df = pd.read_excel("Book1.xlsx")

    # output: y
    y = df['output']

    # inputs: X
    x1 = df['input1']
    x1 = np.asmatrix(x1).T
    x2 = df['input2']
    x2 = np.asmatrix(x2).T
    x3 = df['input3']
    x3 = np.asmatrix(x3).T
    x  = np.concatenate((x1, x2, x3), axis=1)

    # contextual Variable: z
    z = df['z_var']