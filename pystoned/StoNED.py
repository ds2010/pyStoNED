# import dependencies
from . import CNLS
import numpy as np
import math
import scipy.stats as stats
import scipy.optimize as opt
from .constant import CET_ADDI, CET_MULT, FUN_PROD, FUN_COST, RTS_CRS, RTS_VRS


class StoNED(CNLS.CNLS):
    """Stochastic nonparametric envelopment of data (StoNED)
    """

    def __init__(self, y, x, z=None, cet=CET_ADDI, fun=FUN_PROD, rts=RTS_VRS):
        """StoNED model

        Args:
            y (float): output variable. 
            x (float): input variables.
            z (float, optional): Contextual variable(s). Defaults to None.
            cet (String, optional): CET_ADDI (additive composite error term) or CET_MULT (multiplicative composite error term). Defaults to CET_ADDI.
            fun (String, optional): FUN_PROD (production frontier) or FUN_COST (cost frontier). Defaults to FUN_PROD.
            rts (String, optional): RTS_VRS (variable returns to scale) or RTS_CRS (constant returns to scale). Defaults to RTS_VRS.
        """
        super().__init__(y, x, z, cet, fun, rts)

    def get_unconditional_expected_inefficiency(self, method='MOM'):
        # method  = "MOM" : Method of moments
        #         = "QLE" : Quassi-likelihood estimation
        #         = "KDE" : kernel deconvolution estimation
        if self.optimization_status == 0:
            print("Model isn't optimized. Use optimize() method to estimate the model.")
            return False
        if method == "MOM":
            self.__method_of_moment(self.get_residual())
        elif method == "QLE":
            self.__quassi_likelihood(self.get_residual())
        elif method == "KDE":
            self.__gaussian_kernel_estimation(self.get_residual())
        else:
            # TODO(error/warning handling): Raise error while undefined method
            return False
        return self.mu

    def get_technical_inefficiency(self, method='MOM'):
        # method  = "MOM" : Method of moments
        #         = "QLE" : Quassi-likelihood estimation

        # calculate sigma_u, sigma_v, mu, and epsilon value
        if self.optimization_status == 0:
            print("Model isn't optimized. Use optimize() method to estimate the model.")
            return False
        self.get_unconditional_expected_inefficiency(method)
        sigma = self.sigma_u * self.sigma_v / math.sqrt(self.sigma_u ** 2 +
                                                        self.sigma_v ** 2)
        mu = self.epsilon * self.sigma_u / (
            self.sigma_v * math.sqrt(self.sigma_u ** 2 + self.sigma_v ** 2))

        if self.fun == FUN_PROD:
            Eu = sigma * ((stats.norm.pdf(mu) /
                           (1 - stats.norm.cdf(mu) + 0.000001)) - mu)
            if self.cet == CET_ADDI:
                return (self.y - self.get_residual() + self.mu -
                        Eu) / (self.y - self.get_residual() + self.mu)
            elif self.cet == CET_MULT:
                return np.exp(-Eu)
        elif self.fun == FUN_COST:
            Eu = sigma * ((stats.norm.pdf(mu) /
                           (1 - stats.norm.cdf(-mu) + 0.000001)) + mu)
            if self.cet == CET_ADDI:
                return (self.y - self.get_residual() - self.mu +
                        Eu) / (self.y - self.get_residual() - self.mu)
            elif self.cet == CET_MULT:
                return np.exp(Eu)
        # TODO(error/warning handling): Raise error while undefined fun/cet
        return False

    def __method_of_moment(self, residual):
        """Method of moment"""
        residual_mean = np.mean(residual)
        M2 = (residual - residual_mean) ** 2
        M3 = (residual - residual_mean) ** 3

        M2_mean = np.mean(M2, axis=0)
        M3_mean = np.mean(M3, axis=0)

        if self.fun == FUN_PROD:
            if M3_mean > 0:
                M3_mean = 0.0
            self.sigma_u = (M3_mean / ((2 / math.pi) ** (1 / 2) *
                                       (1 - 4 / math.pi))) ** (1 / 3)

        elif self.fun == FUN_COST:
            if M3_mean < 0:
                M3_mean = 0.00001
            self.sigma_u = (-M3_mean / ((2 / math.pi) ** (1 / 2) *
                                        (1 - 4 / math.pi))) ** (1 / 3)

        else:
            # TODO(error/warning handling): Raise error while undefined fun
            return False

        self.sigma_v = (M2_mean -
                        ((math.pi - 2) / math.pi) * self.sigma_u ** 2) ** (1 / 2)
        self.mu = (self.sigma_u ** 2 * 2 / math.pi) ** (1 / 2)
        if self.fun == FUN_PROD:
            self.epsilon = residual - self.mu
        else:
            self.epsilon = residual + self.mu

    def __quassi_likelihood(self, residual):
        def __quassi_likelihood_estimation(lamda, eps):
            """ This function computes the negative of the log likelihood function
            given parameter (lambda) and residual (eps).

            Args:
                lamda (float): signal-to-noise ratio
                eps (list): values of the residual

            Returns:
                float: -logl, negative value of log likelihood
            """
            # sigma Eq. (3.26) in Johnson and Kuosmanen (2015)
            sigma = np.sqrt(
                np.mean(eps ** 2) / (1 - 2 * lamda ** 2 / (math.pi *
                                                           (1 + lamda ** 2))))

            # bias adjusted residuals Eq. (3.25)
            # mean
            mu = math.sqrt(
                2 / math.pi) * sigma * lamda / math.sqrt(1 + lamda ** 2)

            # adj. res.
            epsilon = eps - mu

            # log-likelihood function Eq. (3.24)
            pn = stats.norm.cdf(-epsilon * lamda / sigma)
            return -(-len(epsilon) * math.log(sigma) + np.sum(np.log(pn)) -
                     0.5 * np.sum(epsilon ** 2) / sigma ** 2)

        if self.fun == FUN_PROD:
            lamda = opt.minimize(__quassi_likelihood_estimation,
                                 1.0,
                                 residual,
                                 method='BFGS').x[0]
        elif self.fun == FUN_COST:
            lamda = opt.minimize(__quassi_likelihood_estimation,
                                 1.0,
                                 -residual,
                                 method='BFGS').x[0]
        else:
            # TODO(error/warning handling): Raise error while undefined fun
            return False
        # use estimate of lambda to calculate sigma Eq. (3.26) in Johnson and Kuosmanen (2015)
        sigma = math.sqrt(
            np.mean(residual ** 2) / (1 - (2 * lamda ** 2) / (math.pi *
                                                              (1 + lamda ** 2))))

        # calculate bias correction
        # (unconditional) mean
        self.mu = math.sqrt(2) * sigma * lamda / math.sqrt(math.pi *
                                                           (1 + lamda ** 2))

        # calculate sigma.u and sigma.v
        self.sigma_v = (sigma ** 2 / (1 + lamda ** 2)) ** (1 / 2)
        self.sigma_u = self.sigma_v * lamda

        if self.fun == FUN_PROD:
            self.epsilon = residual - self.mu
        elif self.fun == FUN_COST:
            self.epsilon = residual + self.mu

    def __gaussian_kernel_estimation(self, residual):
        def __gaussian_kernel_estimator(g):
            """Gaussian kernel estimator"""
            return (1 / math.sqrt(2 * math.pi)) * np.exp(-0.5 * g ** 2)

        x = np.array(residual)

        # choose a bandwidth (rule-of-thumb, Eq. (3.29) in Silverman (1986))
        if np.std(x, ddof=1) < stats.iqr(x, interpolation='midpoint'):
            estimated_sigma = np.std(x, ddof=1)
        else:
            estimated_sigma = stats.iqr(x, interpolation='midpoint')
        h = 1.06 * estimated_sigma * len(self.y) ** (-1 / 5)

        # kernel matrix
        kernel_matrix = np.zeros((len(self.y), len(self.y)))
        for i in range(len(self.y)):
            kernel_matrix[i] = np.array([
                __gaussian_kernel_estimator(g=((x[i] - x[j]) / h)) /
                (len(self.y) * h) for j in range(len(self.y))
            ])

        # kernel density value
        kernel_density_value = np.sum(kernel_matrix, axis=0)

        # unconditional expected inefficiency mu
        derivative = np.zeros(len(self.y))
        for i in range(len(self.y) - 1):
            derivative[i +
                       1] = 0.2 * (kernel_density_value[i + 1] -
                                   kernel_density_value[i]) / (x[i + 1] - x[i])

        # expected inefficiency mu
        self.mu = -np.max(derivative)
        if self.fun == FUN_COST:
            self.mu *= -1
