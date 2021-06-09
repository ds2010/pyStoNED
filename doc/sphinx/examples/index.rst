=========
Examples
=========

Consider a standard multivariate, cross-sectional model in production economics:

.. math::
   :nowrap:

   \begin{align}
      y_i & = f(\boldsymbol{x}_i) + \varepsilon_i  \\
          & = f(\boldsymbol{x}_i) + v_i - u_i  \forall i \notag
   \end{align}

where :math:`y_i` is the output of unit :math:`i`, :math:`f: R_+^m \rightarrow R_+` is the production function (cost function) that characterizes the production technology (cost technology),
and :math:`\boldsymbol{x}_i = (x_{i1}, x_{i2}, \cdots, x_{im})^{'}` denotes the input vector of unit :math:`i`. Similar to the literature in Stochastic Frontier analysis (SFA), 
the presented composite error term :math:`\varepsilon_i = v_i - u_i` consists of the inefficiency term :math:`u_i>0` and stochastic noise 
term :math:`v_i`. Different estimation methods can be classified according to the specification of :math:`f` and error term :math:`\varepsilon`
(see Table 1 in Kuosmanen and Johnson, 2010).

Convex Nonparametric Least Square
----------------------------------

.. toctree::
   :maxdepth: 1

   CNLS/additive_prod
   CNLS/multiplicative_prod
   CNLS/additive_cost
   CNLS/multiplicative_cost
   CNLS/corrected_cnls


Stochastic Nonparametric Envelopment of Data
--------------------------------------------

.. toctree::
   :maxdepth: 1

   StoNED/mom
   StoNED/qle
   StoNED/kde
   StoNED/Jondrow


Convex Quantile Approaches
--------------------------
.. toctree::
   :maxdepth: 1
   
   quantile/cqer


Contextual Variables
---------------------

.. toctree::
   :maxdepth: 1

   z/stoned_z


Multiple Outputs (DDF Formulation)
-----------------------------------

.. toctree::
   :maxdepth: 1

   DDF/cnls_ddf
   DDF/cqer_ddf


Monotonic Models
--------------------
.. toctree::
   :maxdepth: 1

   Monotonic/icnls
   Monotonic/icqer


CNLS-G Algorithm (for large sample)
------------------------------------
.. toctree::
   :maxdepth: 1

   CNLSG/cnls_g


Plot the frontier/estimation function
--------------------------------------
.. toctree::
   :maxdepth: 1

   Plot/2d_plot
   Plot/3d_plot


Data Envelopment Analysis
--------------------------
.. toctree::
   :maxdepth: 1

   DEA/dea_io
   DEA/dea_oo
   DEA/dea_ddf
   DEA/dea_und
   DEA/dea_ref


Free Disposal Hull
-------------------
.. toctree::
   :maxdepth: 1

   FDH/fdh_io
   FDH/fdh_oo
