from . import testfunctions as tfs
import numpy as np

from typing import List

class Simulator():
    def __init__(
        self, 
        true_params: List = [1.5], 
        variable_params: np.ndarray = None, 
        include_bias: bool = True, 
        set_seed: int = 2022
    ):
        """Method representing a huge simulation, eg weather forecast, which accepts 
        s variable parameters and d calibration parameters.

        Parameters
        ----------
        true_params : List
            List of true values for calibration parameters.
        variable_params : np.ndarray (optional)
            2D array of size (s, p):
                s is number of 'output locations' requested,
                p is number of variable parameters.
            Default: None
        include_bias : bool
            Toggle to include bias in the observations.
            Default: True
        """

        self.variable_params = variable_params
        self.true_params = true_params
        self.include_bias = include_bias

        # Gaussian bias, not accounted for in simulation model.
        self.bias_mean = 3
        self.bias_var = 2

        # independent Gaussian observation noise
        self.obs_mean = 0
        self.obs_var = 5 # was 0.1

        self.observations = None
        self.errors = None

        np.random.seed(set_seed)


    def run(
        self, 
        calibration_params: np.ndarray, 
        variable_params: np.ndarray = None
    ) -> np.ndarray:
        """Method representing a huge simulation, eg weather forecast, which accepts 
        s variable parameters and d calibration parameters.

        Parameters
        ----------
        calibration_params : np.ndarray
            2D array of size (N, d):
                N is number of simulations to run, 
                d is number of calibration parameters.
        variable_params : np.ndarray (optional)
            2D array of size (s, p):
                s is number of 'output locations' requested,
                p is number of variable parameters.
            Deafult: None

        Returns
        -------
        np.ndarray 
            2D array of size (N*s, p+d+1) where each row corresponds to a simulation 
            output. The first p columns are the variable parameter values, the next 
            d columns are the calibration parameter values and the final column is 
            the simulation output.
        """ 

        assert isinstance(calibration_params, np.ndarray), "calibration_params must be a numpy array!"
        assert len(calibration_params.shape) == 2, "calibration_params must be 2D, {}D array provided".format(len(variable_params.shape))

        variable_params = self.__get_variable_params(variable_params)

        N = calibration_params.shape[0]
        d = calibration_params.shape[1]
        s = variable_params.shape[0]
        p = variable_params.shape[1]

        assert d == len(self.true_params), "Number of calibration parameters, {}, must match number of true parameters, {}.".format(d, len(self.true_params))

        output = np.zeros((N*s, p+d+1))
        for i in range(N):
            calibration_param_row = calibration_params[i, :]

            output[(i*s):((i+1)*s), 0:p] = variable_params
            output[(i*s):((i+1)*s), p:(p+d)] = np.array([calibration_param_row]*d).reshape(-1, d)

            output[i*s:(i+1)*s, -1] = self.__simulation(variable_params, calibration_param_row)
        
        return output

    
    def __get_variable_params(self, variable_params) -> np.ndarray:
        if variable_params is None:
            if self.variable_params is None:
                raise TypeError("variable_params required but not provided.")

            variable_params = self.variable_params

        assert isinstance(variable_params, np.ndarray), "variable_params must be a numpy array!"
        assert len(variable_params.shape)==2, "variable_params must be 2D, {}D array provided".format(len(variable_params.shape))

        return variable_params


    def __simulation(
        self, 
        variable_params: np.ndarray, 
        calibration_param: float
    ) -> np.ndarray:
        # x =         variable_params[:, 0]
        # y =         variable_params[:, 1]
        xy =        variable_params[:, 0:2]
        t_month =   variable_params[:, 2]
        t_hour =    variable_params[:, 3]

        # monthly variation
        rastrigin = tfs.Rastrigin(10)
        annual_field = rastrigin.eval(xy)
        monthly_fluctuation = (1-0.5*np.cos(2*np.pi*(t_month-1)/12))

        # hourly variation
        ackley = tfs.Ackley()
        day_field = ackley.eval(xy)
        hourly_fluctuation = (1-0.5*np.cos(2*np.pi*(t_hour-1)/24))

        # calibration variation
        booth = tfs.Booth()
        calibration_field = booth.eval(xy)
        # k = calibration_param
        # k = 1 + (np.sin(3*calibration_param) + np.sin(10*calibration_param) + np.sin(30*calibration_param))/3
        k = 1.5 + 0.3*calibration_param**3 - calibration_param

        return annual_field*monthly_fluctuation + day_field*hourly_fluctuation - 0.2*k*calibration_field


    def __bias(self, variable_params: np.ndarray) -> np.ndarray:
        # def normal_dist(x, mean, var):
            # return 1/np.sqrt(2*np.pi*var) * np.exp(-0.5*((x - mean)/np.sqrt(var))**2)
        
        xy = variable_params[:, 0:2]

        himmelblau = tfs.Himmelblau()
        return 0.15*himmelblau.eval(xy)


    def __errors(self, n: int) -> np.ndarray:
        if self.errors is None:
            self.errors = np.random.normal(self.obs_mean, np.sqrt(self.obs_var), n)
        
        return self.errors


    def __make_observations(self, variable_params) -> np.ndarray:
        observations = self.run(
            np.array(self.true_params).reshape(1, -1), 
            variable_params=variable_params
        )[:,-1] 
        observations += self.__errors(variable_params.shape[0])
        if self.include_bias == True:
            observations += self.__bias(variable_params)

        return np.array(observations).reshape(-1, 1)
        

    def get_observations(self, variable_params: np.ndarray = None):
        """Returns s (mock) observations, eg weather station measurements, parameterised 
        by p variable parameters.

        Parameters
        ----------
        variable_params : np.ndarray (optional)
            2D array of size (s, p):
                s is number of 'output locations' requested,
                p is number of variable parameters.
            Default: None

        Returns
        -------
        np.ndarray 
            2D array of size (s, p+1) where each row corresponds to a observation. 
            The first p columns are the variable parameter values and the final 
            column is the (mock) observation.
        """

        variable_params = self.__get_variable_params(variable_params)

        observations = self.__make_observations(variable_params)

        return np.hstack((variable_params, observations))


    def reset_observations(self) -> None:
        """Resets the saved (observation) errors. 
        Method call is required when generating more/new observations.
        """

        self.errors = None