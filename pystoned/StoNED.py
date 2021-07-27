# import dependencies
import numpy as np
import math
import scipy.stats as stats
import scipy.optimize as opt
from .utils import tools
from .constant import CET_ADDI, CET_MULT, FUN_PROD, FUN_COST, RED_MOM, RED_QLE, RED_KDE


class StoNED:
    """Stochastic nonparametric envelopment of data (StoNED)
    """

    def __init__(self, model):
        """StoNED
        model: The input model for residual decomposition
        """
        self.model = model
        self.x = model.x

        # If the model is a directional distance based, set cet to CET_ADDI
        if hasattr(self.model, 'gx'):
            self.model.cet = CET_ADDI
            self.y = np.diag(np.tensordot(
                self.model.y, self.model.get_gamma(), axes=([1], [1])))
        else:
            self.y = self.model.y

    def get_unconditional_expected_inefficiency(self, method=RED_MOM):
        """
        Args:
            method (String, optional): RED_MOM (Method of moments) or RED_QLE (Quassi-likelihood estimation) or RED_KDE (Kernel deconvolution estimation). Defaults to RED_MOM.
        """
        tools.assert_optimized(self.model.optimization_status)
        if method == RED_MOM:
            self.__method_of_moment(self.model.get_residual())
        elif method == RED_QLE:
            self.__quassi_likelihood(self.model.get_residual())
        elif method == RED_KDE:
            self.__gaussian_kernel_estimation(self.model.get_residual())
        else:
            raise ValueError("Undefined estimation technique.")
        return self.mu

    def get_technical_inefficiency(self, method=RED_MOM):
        """
        Args:
            method (String, optional): RED_MOM (Method of moments) or RED_QLE (Quassi-likelihood estimation). Defaults to RED_MOM.

        calculate sigma_u, sigma_v, mu, and epsilon value
        """
        tools.assert_optimized(self.model.optimization_status)
        self.get_unconditional_expected_inefficiency(method)
        sigma = self.sigma_u * self.sigma_v / math.sqrt(self.sigma_u ** 2 +
                                                        self.sigma_v ** 2)
        mu = self.epsilon * self.sigma_u / (
            self.sigma_v * math.sqrt(self.sigma_u ** 2 + self.sigma_v ** 2))

        if self.model.fun == FUN_PROD:
            Eu = sigma * ((stats.norm.pdf(mu) /
                           (1 - stats.norm.cdf(mu) + 0.000001)) - mu)
            if self.model.cet == CET_ADDI:
                return (self.y - Eu) / self.y
            elif self.model.cet == CET_MULT:
                return np.exp(-Eu)
        elif self.model.fun == FUN_COST:
            Eu = sigma * ((stats.norm.pdf(mu) /
                           (1 - stats.norm.cdf(-mu) + 0.000001)) + mu)
            if self.model.cet == CET_ADDI:
                return (self.y + Eu) / self.y
            elif self.model.cet == CET_MULT:
                return np.exp(Eu)
        raise ValueError("Undefined model parameters.")

    def __method_of_moment(self, residual):
        """Method of moment"""
        M2 = (residual - np.mean(residual)) ** 2
        M3 = (residual - np.mean(residual)) ** 3

        M2_mean = np.mean(M2, axis=0)
        M3_mean = np.mean(M3, axis=0)

        if self.model.fun == FUN_PROD:
            if M3_mean > 0:
                M3_mean = 0.0
            self.sigma_u = (M3_mean / ((2 / math.pi) ** (1 / 2) *
                                       (1 - 4 / math.pi))) ** (1 / 3)

        elif self.model.fun == FUN_COST:
            if M3_mean < 0:
                M3_mean = 0.00001
            self.sigma_u = (-M3_mean / ((2 / math.pi) ** (1 / 2) *
                                        (1 - 4 / math.pi))) ** (1 / 3)

        else:
            raise ValueError("Undefined model parameters.")

        self.sigma_v = (M2_mean -
                        ((math.pi - 2) / math.pi) * self.sigma_u ** 2) ** (1 / 2)
        self.mu = (self.sigma_u ** 2 * 2 / math.pi) ** (1 / 2)
        if self.model.fun == FUN_PROD:
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

        if self.model.fun == FUN_PROD:
            lamda = opt.minimize(__quassi_likelihood_estimation,
                                 1.0,
                                 residual,
                                 method='BFGS').x[0]
        elif self.model.fun == FUN_COST:
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

        if self.model.fun == FUN_PROD:
            self.epsilon = residual - self.mu
        elif self.model.fun == FUN_COST:
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
        if self.model.fun == FUN_COST:
            self.mu *= -1

    def get_frontier(self, method=RED_MOM):
        """
        Args:
            method (String, optional): RED_MOM (Method of moments) or RED_QLE (Quassi-likelihood estimation). Defaults to RED_MOM.

        calculate sigma_u, sigma_v, mu, and epsilon value
        """
        tools.assert_optimized(self.model.optimization_status)
        self.get_unconditional_expected_inefficiency(method)

        if self.model.fun == FUN_PROD:
            if self.model.cet == CET_ADDI:
                return (self.y - self.model.get_residual()) + self.sigma_u * math.sqrt(2 / math.pi)
            elif self.model.cet == CET_MULT:
                return (self.y / np.exp(self.model.get_residual())) * np.exp(self.sigma_u * math.sqrt(2 / math.pi))
        elif self.model.fun == FUN_COST:
            if self.model.cet == CET_ADDI:
                return (self.y - self.model.get_residual()) - self.sigma_u * math.sqrt(2 / math.pi)
            elif self.model.cet == CET_MULT:
                return (self.y / np.exp(self.model.get_residual())) * np.exp(-self.sigma_u * math.sqrt(2 / math.pi))

        raise ValueError("Undefined model parameters.")
