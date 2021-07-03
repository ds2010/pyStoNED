Estimating firm-specific inefficiencies (JLMS estimator)
==========================================================

After estimating the expected inefficiency :math:`\mu` using methods of moment (MOM) or quasi-likelihood estimation (QLE), [1]_ 
we then employ JLMS estimator proposed by Jondrow et al. (1982) to estimate the firm-specific inefficiencies (Johnson and Kuosmanen, 2015). 
Under the assumption of a normally distributed error term and a half-normally distributed inefficiency term, JLMS formulates the 
conditional distribution of inefficiency :math:`u_i`, given :math:`\varepsilon_i`, and propose the inefficiency estimator as the 
conditional mean :math:`E[u_i|\varepsilon_i]`.

Following Kumbhakar & Lovell (2000), the conditional expected value of inefficiency :math:`E[u_i|\varepsilon_i]` 
for production function and cost function are shown as follow, respectively:

* Production function
    
.. math::
    :nowrap:

    \begin{align*}
        E[u_i \mid \varepsilon_i]
        &= \mu_{*i} + \sigma_* \Bigg[ \frac{\phi(-\mu_{*i}/\sigma_*)}{1-\Phi(-\mu_{*i}/\sigma_*)} \Bigg] \\
        &= \sigma_* \Bigg[ \frac{\phi(\varepsilon_i \lambda/\sigma)}{1-\Phi(\varepsilon_i \lambda/\sigma)} - \frac{\varepsilon_i \lambda}{\sigma} \Bigg].
    \end{align*}
        
where :math:`\mu_*= -\varepsilon \sigma_u^2/\sigma^2`, :math:`\sigma_*^2 = \sigma_u^2\sigma_v^2/\sigma^2`, 
:math:`\lambda = \sigma_u/\sigma_v`, and :math:`\sigma^2 = \sigma_u^2 +\sigma_v^2`. The symbol :math:`\phi` is 
the standard normal density function, and the symbol :math:`\Phi` denotes the cumulative distribution 
function of the standard normal distribution.
    
* Cost function
    
.. math::
    :nowrap:

    \begin{align*}
        E[u_i \mid \varepsilon_i]&= \mu_{*i} + \sigma_* \Bigg[ \frac{\phi(-\mu_{*i}/\sigma_*)}{1-\Phi(-\mu_{*i}/\sigma_*)} \Bigg] \\
        &= \sigma_* \Bigg[ \frac{\phi(\varepsilon_i \lambda/\sigma)}{1-\Phi(-\varepsilon_i \lambda/\sigma)} + \frac{\varepsilon_i \lambda}{\sigma} \Bigg].
    \end{align*}

where :math:`\mu_*= \varepsilon \sigma_u^2/\sigma^2`, :math:`\sigma_*^2 = \sigma_u^2\sigma_v^2/\sigma^2`, 
:math:`\lambda = \sigma_u/\sigma_v`, and :math:`\sigma^2 = \sigma_u^2 +\sigma_v^2`.

The firm-level technical efficiency (TE) is then measured based on the estimated conditional mean. For different model, the technical efficiency is calculated as 

- Production function
    
    - Multiplicative model: :math:`\text{TE} = \exp(-E[u_i \mid  \varepsilon_i])` 
    - Additive model: :math:`\text{TE} = \frac{y - E[u_i \mid  \varepsilon_i]}{y}`

- Cost function

    - Multiplicative model: :math:`\text{TE} = \exp(E[u_i \mid  \varepsilon_i])`
    - Additive model: :math:`\text{TE} = \frac{y+ E[u_i \mid  \varepsilon_i]}{y}`


.. [1] For the expected inefficiency $\mu$ estimated by kernel deconvolution, Dai (2016) proposes a non-parametric strategy where the Richardson–Lucy blind deconvolution algorithm is used to identify firm-specific inefficiencies. However, the `pyStoNED` package only supports the parametric estimation of firm-specific inefficiencies due to the fact that the parametric method is more widely used in efficiency analysis literature.