import numpy as np
from scipy.stats import norm, beta, gamma

from mcmc.data import Data
from .base import BaseModel, Parameter
from mcmc.kernels import log_RBF

class Model(BaseModel):
    """
    """
    _accepted_params = {
        'beta1',
        'beta2',
        'theta',
        'rho'
    }
    _accepted_hyperparams = {
        'l_c1_x',
        'l_c1_t',
        'sigma_c1',
        'l_c2_x',
        'sigma_c2',
        'lambda'
    }


    def __init__(
        self, 
        params: dict, 
        hyperparams: dict, 
        *args, 
        **kwargs
    ):
        """
        """
        super().__init__(params, hyperparams, *args, **kwargs)

    
    def prepare_for_mcmc(self, data: Data) -> None:
        """Prepares the model for MCMC by calculating all the requisite intermediary steps
        for calculating m_d and V_d.
        
        Place ALL of your custom methods here which are used to calculate m_d and V_D.
        They must appear in the correct order.
        """
        # Place ALL of your custom methods here which are used to calculate m_d and V_D.
        # They must appear in the correct order.
        self.calc_H1D1(data)
        self.calc_H1D2(data)
        self.calc_H2D2(data)
        self.calc_H(data)
        self.calc_m_d(data)

        self.calc_V1D1(data)
        self.calc_V1D2(data)
        self.calc_C1D1D2(data)
        self.calc_V2D2(data)
        self.calc_V_d(data)

        self.calc_logpost(data)


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
        super().update(param, index, new_value, data)

        ############################################################

        if param.name == "beta1":
            self.calc_m_d(data)
        elif param.name == "beta2":
            self.calc_m_d(data)
        elif param.name == "theta":
            self.calc_C1D1D2(data)
            self.calc_V_d(data)
        elif param.name == "rho":
            self.calc_H(data)
            self.calc_m_d(data)
            self.calc_V_d(data)
        elif param.name == "l_c1_x":
            self.calc_V1D1(data)
            self.calc_V1D2(data)
            self.calc_C1D1D2(data)
            self.calc_V_d(data)
        elif param.name == "l_c1_t":
            self.calc_V1D1(data)
            self.calc_V1D2(data)
            self.calc_C1D1D2(data)
            self.calc_V_d(data)
        elif param.name == "sigma_c1":
            self.calc_V1D1(data)
            self.calc_V1D2(data)
            self.calc_C1D1D2(data)
            self.calc_V_d(data)
        elif param.name == "l_c2_x":
            self.calc_V2D2(data)
            self.calc_V_d(data)
        elif param.name == "sigma_c2":
            self.calc_V2D2(data)
            self.calc_V_d(data)
        elif param.name == "lambda":
            self.calc_V_d(data)

        ############################################################

        #DO NOT DELETE THIS LINE
        self.calc_logpost(data)


    def calc_prior(self, param: Parameter):
        """Evaluates the prior for the given parameter using the object's attribute value."""

        ############################################################

        if param.name == "beta1":
            # Gaussian(0.5, 10)
            mu = [0.5]
            var = np.diag([10])

            #This is an iterable, can use `next()`
            for index, value in enumerate(param):
                param.prior_densities[index] = norm.logpdf(value, loc=mu, scale=var)
                # self._prior[name][item.index] = norm.logpdf(item.value, loc=mu, scale=var)
        elif param.name == "beta2":
            # Gaussian(0, 100)
            mu = [0]
            var = np.diag([100])
            for index, value in enumerate(param):
                param.prior_densities[index] = norm.logpdf(value, loc=mu, scale=var)
        elif param.name == "theta":
            # Gaussian(0.5, 3)
            mu = [0.5]
            var = np.diag([3])
            for index, value in enumerate(param):
                param.prior_densities[index] = norm.logpdf(value, loc=mu, scale=var)
        elif param.name == "rho":
            # Gaussian(1, 1)
            mu = [1]
            var = np.diag([1])
            for index, value in enumerate(param):
                param.prior_densities[index] = norm.logpdf(value, loc=mu, scale=var)
        elif param.name == "l_c1_x":
            # Beta(2,3)
            for index, value in enumerate(param):
                param.prior_densities[index] = beta.logpdf(value, 2, 3, loc=0, scale=1)
        elif param.name == "l_c1_t":
            # Beta(2,3)
            for index, value in enumerate(param):
                param.prior_densities[index] = beta.logpdf(value, 2, 3, loc=0, scale=1)
        elif param.name == "sigma_c1":
            # Gamma(2, 2)
            for index, value in enumerate(param):
                param.prior_densities[index] = gamma.logpdf(value, a=2, scale=2)
        elif param.name == "l_c2_x":
            # Beta(2,3)
            for index, value in enumerate(param):
                param.prior_densities[index] = beta.logpdf(value, 2, 3, loc=0, scale=1)
        elif param.name == "sigma_c2":
            # Gamma(2, 2)
            for index, value in enumerate(param):
                param.prior_densities[index] = gamma.logpdf(value, a=2, scale=2)
        elif param.name == "lambda":
            # Gamma(0.5, 1)
            for index, value in enumerate(param):
                param.prior_densities[index] = gamma.logpdf(value, a=0.5, scale=1)

        ############################################################

        #DO NOT DELETE THIS LINE
        super().calc_prior(param)


    # def h1_eval(self, x: np.ndarray, y: np.ndarray, theta: np.ndarray, c: np.ndarray):
    #     """Hermite polynonial basis function for regression over emulator."""
        
    #     return np.polynomial.hermite_e.hermeval3d(x, y, theta, c)


    # def h2_eval(self, x: np.ndarray, y: np.ndarray, theta: np.ndarray, c: np.ndarray):
    #     """Hermite polynonial basis function for regression over discrepancy."""
        
    #     return np.polynomial.hermite_e.hermeval3d(x, y, theta, c)


    def calc_H1D1(self, data: Data) -> None:
        """Simulation regression function (H1) at simulation locations (D1)
        """
        self._H1D1 = np.ones(data.y_n)


    def calc_H1D2(self, data: Data) -> None:
        """Simulation regression function (H1) at observation locations (D2)
        """
        self._H1D2 = np.ones(data.z_n)


    def calc_H2D2(self, data: Data) -> None:
        """Discrepancy regression function (H2) at observation locations (D2)
        """
        self._H2D2 = np.ones(data.z_n)


    def calc_H(self, data: Data) -> None:
        """Regression function all data locations.
        """
        self._H = np.zeros((data.y_n + data.z_n, 2))

        self._H[0:data.y_n, 0] = self._H1D1
        self._H[data.y_n:, 0] = self.params['rho'].values*self._H1D2
        self._H[data.y_n:, 1] = self._H2D2

    
    def calc_m_d(self, data: Data) -> None:
        """
        """
        beta1 = self.params['beta1'].values
        beta2 = self.params['beta2'].values
        self.m_d = np.dot(self._H, np.hstack([beta1, beta2]))

        #DO NOT DELETE THIS LINE
        super().calc_m_d(data)


    def calc_V1D1(self, data: Data) -> None:
        """Simulation variance (V1) at simulation locations (D1)
        """
        V1D1 = np.zeros((data.y_n, data.y_n))
        #Create the upper half of the matrix, starting with first row
        for i in range(data.y_n - 1):
            # V1D1[i, (i+1):] = r1_calc(
            #     self.params['l_c1_x'],
            #     data.x_c[i, :].reshape(1, -1), 
            #     data.x_c[(i+1):, :], 
            #     self.params['l_c1_t'],
            #     data.t[i, :].reshape(1, -1), 
            #     data.t[(i+1):, :]
            # )

            #This can be improved. Only need to calculate these everytime the lengthscale parameter changes
            V1D1[i, (i+1):] = np.exp(
                log_RBF(
                    self.params['l_c1_x'].values[0],
                    data.x_c[i, 0],
                    data.x_c[(i+1):, 0]
                ) \
                + log_RBF(
                    self.params['l_c1_x'].values[1],
                    data.x_c[i, 1],
                    data.x_c[(i+1):, 1]
                ) \
                + log_RBF(
                    self.params['l_c1_t'].values[0],
                    data.t[i, 0], 
                    data.t[(i+1):, 0]
                )
            )

            # with open('temp.txt', 'a') as f:
            #     f.write(str(temp))
        V1D1 += V1D1.T
        self._V1D1_unscaled = V1D1 + np.identity(data.y_n)
        self.scale_V1D1()

    
    def scale_V1D1(self) -> None:
        """
        """
        self._V1D1 = self.params['sigma_c1'].values * self._V1D1_unscaled

    
    def calc_V1D2(self, data: Data) -> None:
        """Simulation variance (V1) at observation data locations (D2)
        """
        V1D2 = np.zeros((data.z_n, data.z_n))
        for i in range(data.z_n - 1):
            # V1D2[i, (i+1):] = r1_calc(
            #     self.params['l_c1_x'],
            #     data.x_f[i, :].reshape(1, -1), 
            #     data.x_f[(i+1):, :]
            # )
            V1D2[i, (i+1):] = np.exp(
                log_RBF(
                    self.params['l_c1_x'].values[0],
                    data.x_f[i, 0],
                    data.x_f[(i+1):, 0]
                ) \
                + log_RBF(
                    self.params['l_c1_x'].values[1],
                    data.x_f[i, 1],
                    data.x_f[(i+1):, 1]
                )
            )
        V1D2 += V1D2.T
        self._V1D2_unscaled = V1D2 + np.identity(data.z_n)
        self.scale_V1D2()


    def scale_V1D2(self) -> None:
        """
        """
        self._V1D2 = self.params['sigma_c1'].values * self._V1D2_unscaled


    def calc_C1D1D2(self, data: Data) -> None:
        """Simulation variance (V1) between simulation and observation data locations (D1D2)
        """
        # theta = self.params['theta'].values

        C1D1D2 = np.zeros((data.z_n, data.y_n))
        for i in range(data.z_n):
            # C1D1D2[i, :] = r1_calc(
            #     self.params['l_c1_x'],
            #     data.x_f[i, :].reshape(1, -1),
            #     data.x_c[:, :],
            #     self.params['l_c1_t'],
            #     theta.reshape(1, -1),
            #     data.t[:, :]
            # )

            #The slightly different form from eg V1D1() is correct. 
            #We're not constructing a symetric (sub)matrix.
            C1D1D2[i, :] = np.exp(
                log_RBF(
                    self.params['l_c1_x'].values[0],
                    data.x_f[i, 0],
                    data.x_c[:, 0]
                ) \
                + log_RBF(
                    self.params['l_c1_x'].values[1],
                    data.x_f[i, 1],
                    data.x_c[:, 1]
                ) \
                + log_RBF(
                    self.params['l_c1_t'].values[0],
                    self.params['theta'].values[0], 
                    data.t[:, 0]
                )
            )
        self._C1D1D2_unscaled = C1D1D2
        self.scale_C1D1D2()

    
    def scale_C1D1D2(self) -> None:
        """
        """
        self._C1D1D2 = self.params['sigma_c1'].values * self._C1D1D2_unscaled


    def calc_V2D2(self, data:Data) -> None:
        """Discrepancy variance (V2) at observation data locations (D2)
        """
        V2D2 = np.zeros((data.z_n, data.z_n))
        for i in range(data.z_n - 1):
            # V2D2[i, (i+1):] = r2_calc(
            #     self.params['l_c2_x'],
            #     data.x_f[i, :].reshape(1, -1), 
            #     data.x_f[(i+1):, :]
            # )

            V2D2[i, (i+1):] = np.exp(
                log_RBF(
                    self.params['l_c2_x'].values[0],
                    data.x_f[i, 0],
                    data.x_f[(i+1):, 0]
                ) \
                + log_RBF(
                    self.params['l_c2_x'].values[1],
                    data.x_f[i, 1],
                    data.x_f[(i+1):, 1]
                )
            )
        V2D2 += V2D2.T
        self._V2D2_unscaled = V2D2 + np.identity(data.z_n)
        self.scale_V2D2()

    
    def scale_V2D2(self) -> None:
        """
        """
        self._V2D2 = self.params['sigma_c2'].values * self._V2D2_unscaled

    
    def calc_V_d(self, data: Data) -> None:
        """
        """
        a = data.y_n
        b = data.z_n

        self.V_d = np.zeros((a+b, a+b))

        rho = self.params['rho'].values
        lambda_ = self.params['lambda'].values

        self.V_d[0:a, 0:a] = self._V1D1
        self.V_d[a:, 0:a] = rho*self._C1D1D2
        self.V_d[0:a, a:] = np.transpose(self.V_d[a:, 0:a])
        self.V_d[a:, a:] = self._V2D2 + self._V1D2*rho**2 + lambda_*np.identity(b)
        
        #DO NOT DELETE THIS LINE
        super().calc_V_d(data)


