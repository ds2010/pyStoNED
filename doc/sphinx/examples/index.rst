=========
Examples
=========

Consider a standard multivariate, cross-sectional model in production economics:

.. math::
   :nowrap:

   \begin{align}
      y_i & = f(\boldsymbol{x}_i) + \varepsilon_i  \\
          & = f(\boldsymbol{x}_i) + v_i - u_i  \quad \forall i \notag
   \end{align}

where :math:`y_i` is the output of unit :math:`i`, :math:`f: R_+^m \rightarrow R_+` is the production function (cost function) that characterizes the production technology (cost technology),
and :math:`\boldsymbol{x}_i = (x_{i1}, x_{i2}, \cdots, x_{im})^{'}` denotes the input vector of unit :math:`i`. Similar to the literature in Stochastic Frontier analysis (SFA), 
the presented composite error term :math:`\varepsilon_i = v_i - u_i` consists of the inefficiency term :math:`u_i>0` and stochastic noise 
term :math:`v_i`. To estimate the function :math:`f`, one could use the parametric and nonparametric methods or neoclassical and frontier models, 
of which methods are classified based on the specification of :math:`f` and error term :math:`\varepsilon` (see \citeauthor{Kuosmanen2010a}, \citeyear{Kuosmanen2010a}). In this paper, 
we assume certain axiomatic properties (e.g., monotonicity, concavity) instead of \textit{a priori} functional form for the function :math:`f` 
and apply the nonparametric methods to estimate the function :math:`f`.



Convex Nonparametric Least Square
----------------------------------

.. toctree::
   :maxdepth: 1

   CNLS/additive
   CNLS/multiplicative
   CNLS/corrected_cnls


Convex Quantile and Expectile Approaches
-----------------------------------------

.. toctree::
   :maxdepth: 1
         
   quantile/cqr
   quantile/cer


Contextual Variables
---------------------

.. toctree::
   :maxdepth: 1

   z/z_variables

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

      
      
Stochastic Nonparametric Envelopment of Data
--------------------------------------------

.. toctree::
   :maxdepth: 1

   StoNED/mom
   StoNED/qle
   StoNED/kde
   StoNED/Jondrow


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
