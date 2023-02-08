from typing import Dict
from abc import ABC, abstractmethod

import numpy as np

from mcmc.parameter import Parameter
from mcmc.data import Data


class BaseModel(ABC):
    """Class representing a model based on the Kennedy & O'Hagan (2001) framework.
    
    Model
    -----
    y_f(x) = rho*eta(x,theta) + delta(x) + e(x)
    """

    _accepted_params = set()
    _accepted_hyperparams = set()


    @abstractmethod
    def __init__(
        self, 
        params: Dict[str, Parameter], 
        hyperparams: Dict[str, Parameter], 
        *args, 
        **kwargs
    ):
        """
        """

        if set(params.keys()) != self._accepted_params:
            raise ValueError(f"Parameter names must match exactly for the chosen model. {self._accepted_params}")
        if set(hyperparams.keys()) != self._accepted_hyperparams:
            raise ValueError(f"Hyperparameter names must match exactly for the chosen model. {self._accepted_hyperparams}")

        self.m_d = None
        self.V_d = None
        self.V_d_chol = None
        self._prior_densities = {}

        #Combine params and hyperparams into a single attribute, params.
        #There is no reason to distinquish between them.
        # #Python 3.9+
        # self.params = params | hyperparams
        #Python 3.5+
        self.params = {**params, **hyperparams}

        self.total_param_count_long = 0
        for param in self.params.values():
            self.total_param_count_long += len(param)

        self.__dict__.update(kwargs)


    @abstractmethod
    def prepare_for_mcmc(self, data: Data) -> None:
        """
        """
        #Calculate priors
        for param in self.params.values():
            self.calc_prior(param)


    def get_param_names(self) -> set:
        """Returns set of parameter names.
        """
        return self._accepted_params.union(self._accepted_hyperparams)

    
    def get_param_long_names(self) -> set:
        """Returns set of parameter names with index numbers appended.
        """
        output = set()
        for name, param in self.params.items():
            output.update([f"{name}_{i}" for i in range(len(param))])
        return output


    @abstractmethod
    def update(
        self, 
        param: Parameter, 
        index: int,
        new_value: float,
        data: Data,
    ) -> None:
        """Updates the parameter and recalculates any necessary components
        """
        param.update(index, new_value)

        self.calc_prior(param)


    @abstractmethod
    def calc_m_d(self, data: Data) -> None:
        """
        """
        pass


    @abstractmethod
    def calc_V_d(self, data: Data) -> None:
        """
        """
        self.V_d_chol = np.linalg.cholesky(self.V_d)


    def calc_loglike(self, data: Data) -> None:
        """
        """
        u = np.linalg.solve(self.V_d_chol, data.d - self.m_d)
        Q = np.dot(u, u.T)
        logdet = np.sum(np.log(np.diag(self.V_d_chol)))
        self.loglike = -logdet - 0.5*Q


    @abstractmethod
    def calc_prior(self, param: Parameter):
        """Evaluates the prior for the given parameter using the object's attribute value."""
        self._prior_densities[param.name] = param.get_prior_density_sum()

    
    def calc_logpost(self, data: Data):
        self.calc_loglike(data)
        self.logpost = np.sum(list(self._prior_densities.values())) + self.loglike


    def get_model_params(self) -> dict:
        """
        """
        output = {}
        for name, param in self.params.items():
            output.update(
                zip(
                    [f"{name}_{i}" for i in range(len(param))],
                    param.values
                )
            )
        return output