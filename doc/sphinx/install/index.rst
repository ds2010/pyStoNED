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

All modules supported by pyStoNED are either additive or multiplicative models, depending on the specification of the error term's structure.
From the perspective of optimization, the additive models are usually the QP problem with an exception of the CQR model, which is a linear programming
(LP) problem, and all multiplicative models are the NLP problem. The attribute of models determines which type of solver will meet our needs, 
i.e., QP/LP-solver (e.g., ``CPLEX``, ``MOSEK``) or NLP-solver (e.g., ``MINOS``, ``KNITRO``). 

`Pyomo <http://www.pyomo.org/>`_ provides a complete application programming interface to import those commercial off-the-shelf solvers. 
pyStoNED interfaces with Pyomo to connect the network-enabled optimization system (`NEOS <https://neos-server.org/neos/>`_) server that 
freely provides a large number of academic solvers for solving the additive or multiplicative models remotely. pyStoNED also provides 
support for ``MOSEK`` through Pyomo to calculate the additive models locally. Specifically,


   * Remote solver

   In pyStoNED, we use the following code to estimate the additive or multiplicative models via the remote solver provided by NEOS.

   ::

      model.optimize('email@address')

   Replace the argument ``email@address`` with your own email address. [2]_  
   By default, the additive and multiplicative models will be solved by ``MOSEK`` and ``KNITRO``, respectively. 
   In addition, the users can freely choose their prefer solver, e.g., ``CPLEX``, using

   ::

      model.optimize(email="email@address", solver='cplex')

   * Local solver

   To solve the additive models with ``MOSEK``, the academic license is required to install on the user's machine in advance.

   ::

      ## MOSEK Optimizer: License installation ##

      1. Requrest Personal Academic License
      
         https://www.mosek.com/license/request/personal-academic/

      2. Save the license file under the user's home directory 
         with a folder named "mosek". e.g.,
         
         Windows: c:\users\xxxx\mosek\mosek.lic
         Unix/OS X: /home/xxxx/mosek/mosek.lic

         Note: xxxx is the User ID on the computer

   After that, we can use the following code to calculate the additive models.

   ::

      model.optimize(OPT_LOCAL)

   The parameter  ``OPT_LOCAL`` is added in the function ``.optimize(...)`` to indicate that the model is computed locally. 
   Similarly, one can use the parameter ``solver`` to select other solver if the corresponding license is available. 


Overall, the remote solver is highly recommended for ALL light computing jobs. The local solver for 
computing the multiplicative model will be supported in pyStoNED when the free and stable NLP solver is available.


.. [1] The GitHub repo provides the latest development version.
.. [2] As of January 2021, NOES requires a valid email address in all submission; see `NEOS Server FAQ <https://neos-guide.org/content/FAQ#email>`_.