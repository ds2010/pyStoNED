JLMS estimator
==================

After estimating the expected inefficiency using methods of moment (MOM) or quasi-likelihood estimation (QLE), 
We can use JLMS estimator proposed by Jondrow et al. (1982) 
to estimate the firm-specific inefficiencies (Johnson & Kuosmanen, 2015). 
Under the assumption of a normally distributed error term and a half-normally 
distributed inefficiency term, they formulate the conditional distribution of 
inefficiency :math:`u_i`, given :math:`\varepsilon_i`, and propose the inefficiency estimator 
as the conditional mean :math:`E(u_i|\varepsilon_i)`.

1. :math:`E(u_i \mid \varepsilon_i)`: Following Kumbhakar & Lovell (2000), the conditional expected value of inefficiency are

    * Production function
    
    .. math::
       :nowrap:

        \begin{align*}
        E(u_i \mid \varepsilon_i)
        &= \mu_{*i} + \sigma_* \Bigg[ \frac{\phi(-\mu_{*i}/\sigma_*)}{1-\Phi(-\mu_{*i}/\sigma_*)} \Bigg] \\
        &= \sigma_* \Bigg[ \frac{\phi(\varepsilon_i \lambda/\sigma)}{1-\Phi(\varepsilon_i \lambda/\sigma)} - \frac{\varepsilon_i \lambda}{\sigma} \Bigg].
        \end{align*}
        
        where $\mu_*= -\varepsilon \sigma_u^2/\sigma^2$ and $\sigma_*^2 = \sigma_u^2\sigma_v^2/\sigma^2$.

    
    * Cost function
    
    .. math::
       :nowrap:

        \begin{align*}
        E(u_i \mid \varepsilon_i)&= \mu_{*i} + \sigma_* \Bigg[ \frac{\phi(-\mu_{*i}/\sigma_*)}{1-\Phi(-\mu_{*i}/\sigma_*)} \Bigg] \\
        &= \sigma_* \Bigg[ \frac{\phi(\varepsilon_i \lambda/\sigma)}{1-\Phi(-\varepsilon_i \lambda/\sigma)} + \frac{\varepsilon_i \lambda}{\sigma} \Bigg].
        \end{align*}

        where $\mu_*= \varepsilon \sigma_u^2/\sigma^2$ and $\sigma_*^2 = \sigma_u^2\sigma_v^2/\sigma^2$.

2. Technical inefficiency (TE)

    - Production function
        - Logged Dependent Variable: :math:`\text{TE} = \text{exp}(-E(u_i \mid \varepsilon_i))` 
        - Otherwise,  :math:`\text{TE} = \frac{Y - E(u_i \mid \varepsilon_i)}{Y}`
            
    - Cost function
        - Logged Dependent Variable: :math:`\text{TE} = \text{exp}(E(u_i \mid \varepsilon_i))`
        - Otherwise,  :math:`\text{TE} = \frac{Y+ E(u_i \mid \varepsilon_i)}{Y}`

References:

[1] Johnson, A. L. & Kuosmanen, T. (2015), An Introduction to CNLS and StoNED Methods for Efficiency Analysis: Economic Insights and Computational Aspects, in S. C. Ray, S. C. Kumbhakar & P. Dua (eds), Benchmarking for Performance Evaluation: A Production Frontier Approach, Springer, chapter 3, pp. 117–186.

[2] Jondrow, J., Lovell, C. A. K., Materov, I. S. & Schmidt, P. (1982), On the estimation of technical inefficiency in the stochastic frontier production function model, Journal of Econometrics 19, 233–238.

[3] Kumbhakar, S. C. & Lovell, C. A. K. (2000), Stochastic Frontier Analysis, Cambridge University Press.


