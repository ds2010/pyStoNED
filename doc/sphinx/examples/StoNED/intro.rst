Combing virtues of SFA and DEA in a unified framework, Stochastic Nonparametric Envelopment 
of Data (StoNED) (kuosmanen, 2006) uses a composed error 
term to model both inefficiency :math:`u` and noise :math:`v` without assuming a 
functional form of :math:`f`. Analogous to the COLS/C :math:`^2` NLS estimators, the StoNED 
estimator consists of the following four steps:

- Step 1: Estimating the conditional mean :math:`E[y_i \,| \, \boldsymbol{x}_i]` using CNLS estimator
- Step 2: Estimating the expected inefficiency :math:`\mu` based on the residual :math:`\varepsilon_i^{CNLS}`
- Step 3: Estimating the StoNED frontier :math:`\hat{f}^{StoNED}` based on the :math:`\hat{\mu}`
- Step 4: Estimating firm-specific inefficiencies :math:`E[u_i \mid \varepsilon_i^{CNLS}]`

Beside the CNLS estimator, we can apply other convex regression approaches such as 
ICNLS and CNLS-DDF to estimate the conditional mean in the first step 
(see Keshvari and Kuosmanen, 2013; Kuosmanen and Johnson, 2017). 
However, the quantile and expectile related estimators introduced in **Examples** 
can not be integrated into StoNED framework at present. 


After obtaining the residuals (e.g., :math:`\hat{\varepsilon}_i^{CNLS}`) from the convex regression approaches, 
one can estimate the expected value of the inefficiency term :math:`\mu = E(u_i)`. In practice, three commonly 
used methods are available to estimate the expected inefficiency :math:`\mu`: method of moments (Aigner et al., 1977), 
quasi-likelihood estimation (Fan et al., 1996), and the kernel deconvolution estimation (Hall and Simar, 2002). 
We will next briefly review these three approaches and focus on demonstrating the application of 
`pyStoNED`; see more detailed theoretical introduction in Kuosmanen et al. (2015).