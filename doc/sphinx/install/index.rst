.. _Installation:

Installation
==============

pyStoNED supports Python 3.8 or later versions on Linux, macOS, and Windows. We can install the package
using pip or GitHub.

PyPI
----
::

   pip install pystoned

GitHub [1]_
-----------
::

   pip install -U git+https://github.com/ds2010/pyStoNED

Solver
------

All models supported by pyStoNED are either additive or multiplicative models, depending on the specification of the error term. 
From the perspective of optimization, the additive models are usually stated as QP problems with an exception of the CQR model, 
which is a linear programming (LP) problem, whereas all multiplicative models are NLP problems. 

To solve the relevant QP or NLP problems, external off-the-shelf solvers are required. In our experience, ``CPLEX`` or ``MOSEK``
provide reliable and convenient platforms for solving the QP and LP problems. The NLP problem can be efficiently solved by ``MINOS`` or ``KNITRO``. 
Note that the tailored algorithm for specific estimators could also be used to solve these models. However, tailored algorithms are not the most convenient choice for our general 
application purposes. With the help of `Pyomo <http://www.pyomo.org/>`_, all models supported by pyStoNED are computable by these off-the-shelf solvers. 

   * Remote solver

   pyStoNED interfaces with Pyomo to access the network-enabled optimization system (`NEOS <https://neos-server.org/neos/>`_) server that 
   freely provides a large number of academic solvers for solving the additive or multiplicative models remotely. In this case, the users do not 
   need to install solvers and corresponding licenses on their own computer. Here we have a model estimated by remote solver

   ::

      model.optimize('email@address')

   Replace the argument ``email@address`` with your email address. [2]_  
   By default, the additive and multiplicative models will be solved by ``MOSEK`` and ``KNITRO``, respectively. In addition, 
   the users can freely choose their preferred solver, e.g., ``MINOS``, using

   ::

      model.optimize(email="email@address", solver='minos')

   * Local solver

   Pyomo also provides an application programming interface for pyStoNED to import the local solvers. In the pyStoNED package, 
   ``MOSEK`` is attached as the internal dependency to solve the additive model. However, the ``MOSEK`` academic license is required to be installed. 
   
   ::

      ## MOSEK Optimizer: License installation ##

      1. Request Personal Academic License
      
         https://www.mosek.com/license/request/personal-academic/

      2. Save the license file under the user's home directory 
         with a folder named "mosek". e.g.,
         
         Windows: c:\users\xxxx\mosek\mosek.lic
         Unix/OS X: /home/xxxx/mosek/mosek.lic

         Note: xxxx is the User ID on the computer   
   
   Otherwise, the following error message will display on the screen when calculating CNLS/StoNED or other variants.
   
   ::

      MOSEK error 1008: License cannot be located. The default search path is ':/home/user/mosek/mosek.lic:'.

   After that, we can use the following code to calculate the additive models.

   ::

      model.optimize(OPT_LOCAL)

   The parameter ``OPT_LOCAL`` is added in the function ``.optimize(...)`` to indicate that the model is computed locally. 
   Similarly, one can use the parameter ``solver`` to select another solver when it is available. 

Overall, the remote solver through the NEOS server is highly recommended for all light computing jobs, where, in general, 
the number of observations is no more than 500. The local solver for calculating the multiplicative model will be supported 
in pyStoNED when a free and stable NLP solver is available.

.. [1] The GitHub repo provides the latest development version.
.. [2] As of January 2021, NOES requires a valid email address in all submission; see `NEOS Server FAQ <https://neos-guide.org/content/FAQ#email>`_.