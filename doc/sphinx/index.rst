.. pyStoNED documentation master file, created by
   sphinx-quickstart on Sun Nov 15 14:11:07 2020.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to pyStoNED 0.4
====================================

`pyStoNED <https://pypi.org/project/pystoned/>`_ is a Python package that provides functions for estimating Convex Nonparametric Least Square 
(`CNLS <https://pubsonline.informs.org/doi/abs/10.1287/opre.1090.0722>`_), Stochastic Nonparametric Envelopment of Data (`StoNED <https://link.springer.com/article/10.1007/s11123-010-0201-3>`_), 
and other various `StoNED`-related variants (e.g., `CQR <https://www.sciencedirect.com/science/article/pii/S0140988320300979>`_, 
`ICNLS <https://www.sciencedirect.com/science/article/abs/pii/S0377221713004748>`_). 
It allows the user to estimate the StoNED-related models in an open-access environment rather than in commercial software (e.g., ``GAMS`` and ``MATLAB``).

For example, the following code estimates the basic CNLS model and plot the production frontier.

.. code:: python

   # import packages
   import numpy as np
   from pystoned import CNLS
   from pystoned.constant import CET_ADDI, FUN_PROD, OPT_LOCAL, RTS_VRS
   
   # set seed
   np.random.seed(0)  
   
   # generat DMUs: DGP
   x = np.sort(np.random.uniform(low=1, high=10, size=50))
   u = np.abs(np.random.normal(loc=0, scale=0.7, size=50))
   y_true = 3 + np.log(x)
   y = y_true - u

   # define the CNLS model
   model = CNLS.CNLS(y, x, z=None, cet = CET_ADDI, fun = FUN_PROD, rts = RTS_VRS)
   # solve the model
   model.optimize(OPT_LOCAL)

   # display the residuals
   model.display_residual()

   # plot CNLS frontier
   model.plot2d(0, fig_name="CNLS frontier")

   
.. toctree::
   :hidden:

   install/index

.. toctree::
   :maxdepth: 3
   :hidden:

   api/index

.. toctree::
   :maxdepth: 3
   :hidden:

   datasets/index  
      
.. toctree::
   :maxdepth: 3
   :hidden:

   examples/index

.. toctree::
   :hidden:

   advanced/index   

.. toctree::
   :hidden:

   contributing/index

.. toctree::
   :hidden:

   citing/index.md

.. toctree::
   :hidden:

   free_course/index

.. toctree::
   :hidden:

   acronyms/index.md

