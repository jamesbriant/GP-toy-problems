from mcmc.data import Data
from mcmc.models.base import BaseModel
from mcmc.parameter import Parameter

import numpy as np
from scipy.stats import uniform, beta, gamma

class Model(BaseModel):
    """
    """

    #ORDERED LIST FOR MCMC
    _accepted_params = [
        "theta",
        "beta_eta",
        "lambda_eta",
        "lambda_epsilon_eta",
        "lambda_delta",
        "lambda_epsilon"
    ]


    def __init__(
        self, 
        params: dict,
        *args, 
        **kwargs
    ):
        """
        """
        #DO NOT DELETE THIS LINE
        super().__init__(params, *args, **kwargs)
    
    def prepare_for_mcmc(self, data: Data) -> None:
        """Prepares the model for MCMC by calculating all the requisite intermediary steps
        for calculating m_d and V_d.
        
        Place ALL of your custom methods here which are used to calculate m_d and V_D.
        They must appear in the correct order.
        """
        ############################################################
        #Place your code here

        # Place ALL of your custom methods here which are used to calculate m_d and V_D.
        # They must appear in the correct order.

        self.D = dist_tensor(data, self.params['theta'])

        self.calc_sigma_eta(data)
        self.calc_sigma_delta(data)
        self.calc_sigma_epsilon(data)
        self.calc_sigma_epsilon_eta(data)

        ############################################################
        #The next line calculates and saves the prior densities.
        #DO NOT DELETE THIS LINE
        super().prepare_for_mcmc(data)


    def update(
        self, 
        param: Parameter, 
        index: int,
        new_value: float,
        data: Data,
    ) -> None:
        """Updates the parameter and recalculates any necessary components
        """
        #DO NOT DELETE THIS LINE
        super().update(param, index, new_value, data)
        ############################################################
        #Place your code here

        if param.name == "theta":
            self.calc_m_d(data)
        elif param.name == "beta_eta":
            self.calc_sigma_eta(data)
            self.calc_V_d(data)
        elif param.name == "lambda_eta":
            self.calc_sigma_eta(data)
            self.calc_V_d(data)
        elif param.name == "lambda_delta":
            self.calc_sigma_delta(data)
            self.calc_V_d(data)
        elif param.name == "lambda_epsilon":
            self.calc_sigma_epsilon(data)
            self.calc_V_d(data)
        elif param.name == "lambda_epsilon_eta":
            self.calc_sigma_epsilon_eta(data)
            self.calc_V_d(data)


        ############################################################
        #DO NOT DELETE THIS LINE
        self.calc_logpost(data)


    def calc_prior(self, param: Parameter):
        """Evaluates the prior for the given parameter using the object's attribute value."""
        ############################################################
        #Place your code here

        if param.name == "theta":
            pass
        elif param.name == "beta_eta":
            rho_eta = np.exp(-param.values/4)
            rho_eta[rho_eta>0.999] = 0.999
            for index, value in enumerate(rho_eta):
                param.prior_densities[index] = beta.logpdf(value, 1, 0.5, loc=0, scale=1)
        # elif param.name == "beta_delta":
        #     rho_delta = np.exp(-param.values/4)
        #     rho_delta[rho_delta>0.999] = 0.999
        #     for index, value in enumerate(rho_delta):
        #         param.prior_densities[index] = beta.logpdf(value, 1, 0.4, loc=0, scale=1)
        elif param.name == "lambda_eta":
            for index, value in enumerate(param):
                param.prior_densities[index] = gamma.logpdf(value, a=10, scale=10)
        elif param.name == "lambda_delta":
            for index, value in enumerate(param):
                param.prior_densities[index] = gamma.logpdf(value, a=10, scale=0.3)
        elif param.name == "lambda_epsilon":
            for index, value in enumerate(param):
                param.prior_densities[index] = gamma.logpdf(value, a=10, scale=0.03)
        elif param.name == "lambda_epsilon_eta":
            for index, value in enumerate(param):
                param.prior_densities[index] = gamma.logpdf(value, a=10, scale=0.001)

        ############################################################

        #DO NOT DELETE THIS LINE
        super().calc_prior(param)


    
    def calc_m_d(self, data: Data) -> None:
        """
        """
        ############################################################
        #Place your code here

        self.m_d = np.zeros((data.n + data.m))

        ############################################################
        #DO NOT DELETE THIS LINE
        super().calc_m_d(data)


    def calc_sigma_eta(self, data: Data) -> None:
        """
        """
        self._sigma_eta = np.sum(
            np.tensordot(
                self.params['beta_eta'].values,
                self.D, 
                axes=0
            ),
            #axis=0 sums over parameters
            #axis=1 sums over vector dot product
            axis=(0,1)
        )
        
    
    def calc_sigma_delta(self, data: Data) -> None:
        """
        """
        # self._sigma_delta = np.sum(
        #     np.tensordot(
        #         self.params['beta_delta'],
        #         self.D[:, :data.n, :data.n], 
        #         axes=0
        #     ), 
        #     #axis=0 sums over parameters
        #     #axis=1 sums over vector dot product
        #     axis=(0, 1)
        # )
        self._sigma_delta = self.params['lambda_delta'].values * np.eye(data.n)


    def calc_sigma_epsilon(self, data: Data) -> None:
        """
        """
        self._sigma_epsilon = self.params['lambda_epsilon'].values * np.eye(data.n)

    
    def calc_sigma_epsilon_eta(self, data: Data) -> None:
        """
        """
        self._sigma_epsilon_eta = self.params['lambda_epsilon_eta'].values * np.eye(data.m)
        

    
    def calc_V_d(self, data: Data) -> None:
        """
        """
        ############################################################
        #Place your code here

        self.V_d = self._sigma_eta
        self.V_d[:data.n, :data.n] += self._sigma_delta + self._sigma_epsilon
        self.V_d[data.n:, data.n:] += self._sigma_epsilon_eta

        ############################################################
        #DO NOT DELETE THIS LINE
        super().calc_V_d(data)


def dist_tensor(data: Data, theta: Parameter) -> np.ndarray:
    D = np.zeros((
        data.p + data.q,
        data.n + data.m,
        data.n + data.m
    ))

    #control variables
    for param_i in range(data.p):
        #Each matrix is symmetric and entries on leading diagonal are all 0.
        #Therefore we only calculate the upper triangular, then add the transpose.

        #analyse observations
        for row_i in range(data.n - 1):
            #observation x observation
            D[param_i, row_i, (row_i+1):data.n] = (data.x_f[row_i, param_i] - data.x_f[(row_i+1):, param_i])**2

            #observation x simulation
            D[param_i, row_i, data.n:] = (data.x_f[row_i, param_i] - data.x_c[:, param_i])**2

        #calculate the final row of observation x simulation
        D[param_i, data.n-1, data.n:] = (data.x_f[data.n-1, param_i] - data.x_c[:, param_i])**2

        #analyse simulations
        for row_i in range(data.m - 1):
            #simulatio x simulation
            D[param_i, data.n+row_i, (data.n+row_i+1):] = (data.x_c[row_i, param_i] - data.x_c[(row_i+1):, param_i])**2

        #Add the transpose
        D[param_i, :, :] += D[param_i, :, :].T

    #calibration variables
    for param_j in range(data.q):
        #Each matrix is symmetric and entries on leading diagonal are all 0.
        #Therefore we only calculate the upper triangular, then add the transpose.

        #analyse observations
        for row_i in range(data.n):
            #observation x observation values are all 0

            #observation x simulation
            D[param_j, row_i, data.n:] = (theta.values[param_j] - data.t[row_i, param_j])**2

        #analyse simulations
        for row_i in range(data.m - 1):
            #simulation x simulation
            D[param_j, data.n+row_i, (data.n+row_i+1):] = (data.t[row_i, param_j] - data.t[(row_i+1):, param_j])**2

        #Add the transpose
        D[param_j, :, :] += D[param_j, :, :].T

    return D