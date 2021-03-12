.. _datasets:

Datasets
===========

In this section, the package provides 4 datasets: First two are used in large number of CNLS/StoNED
liturature; the others are commonly used in the SFA liturature. In the following section (:ref:`Examples`), our tutorials
will resort to these example data.

.. toctree::
    :maxdepth: 2

    firms/index.md
    countries/index.md
    front41Data/index.md
    riceProdPhil/index.md

Import example data from pyStoNED
-----------------------------------------

- Finnish electricity firm data

.. code:: python

    from pystoned.dataset import load_Finnish_electricity_firm

    data = load_Finnish_electricity_firm(x_select=['Energy', 'Length', 'Customers'], 
                                        y_select=['TOTEX'], 
                                        z_select=['PerUndGr'])
                                        
- import OECD GHG emissions data

- import Tim Coelli’s Frontier 4.1 data

- import rice production data


Import personal data
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

