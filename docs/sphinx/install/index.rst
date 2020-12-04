.. _install:

Install
=======

pyStoNED supports Python 3 on Linux, macOS, and Windows. You can use pip or GitHub for installation.

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

To estimate the StoNED-related models, we usually use two types of solvers: quadratic programming solver (QP-solver) and nonlinear programming solver (NLP-solver). In the pyStoNED package, we can import these two kind of solvers locally or remotely. Specifically,

  1. For remote solvers, we usually rely on the solvers provided by the `NEOS server <https://neos-server.org/neos/>`_, where we can choose such solvers as `cplex`, `mosek`, `minos`, and `knitro`. The first two solvers can be used to calculate the LP models, the last two can be used to estimate the NLP models, and the second is more suit for the QP models. However, sometimes, the NEOS server is not so stable, and some unknown errors could appear.

  2. For local solvers, we can use the solvers embedded in GAMS or API solvers (e.g., MOSEK). Compared with the remote solvers, the local solver is pretty robust to performance, but the amount of solvers is limit.

In the pyStoNED, one can free to choose the solvers by changing the argument ``remote=``. For examples:

  * Using local solver
   
   ::

      model.optimize(remote=False)

  * Using remote solver

   ::

      model.optimize(remote=True)

For local solver, the pyStoNED currently only supports the additive model estimation by using the `MOSEK <https://www.mosek.com/>`_. If you do that, you have to install MOSEK API by yourself. The installation tutorial is provided as follows:

::

   ## MOSEK Optimizer API Installation ##

   1. Requrest Personal Academic License MOSEK-Academic License.

   2. Save the license file under the user's home directory with a folder named "mosek". For example:
      Windows: c:\users\xxxx\mosek\mosek.lic
      Unix/OS X: /home/xxxx/mosek/mosek.lic

      Note: xxxx is the User ID on the computer

   3. Install mosek Package
      pip: pip install Mosek
      anconda: conda install -c mosek mosek

   4. Testing the Installation
      start Python and try: import mosek

   Other detailed installation process can be seen in MOSEK docs.


For remote solver, the additive model and multiplicative model are calculated by ``MOSEK`` and ``KNITRO`` on NEOS server, respectively.


.. [1] The GitHub repo provides the latest development version.
