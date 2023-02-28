from mcmc.data import Data
from .base import BaseModel, Parameter

class Model(BaseModel):
    """
    """

    #ORDERED LIST FOR MCMC
    _accepted_params = [
        #define parameter name strings here
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



        ############################################################
        #DO NOT DELETE THIS LINE
        self.calc_logpost(data)


    def calc_prior(self, param: Parameter):
        """Evaluates the prior for the given parameter using the object's attribute value."""
        ############################################################
        #Place your code here



        ############################################################

        #DO NOT DELETE THIS LINE
        super().calc_prior(param)


    
    def calc_m_d(self, data: Data) -> None:
        """
        """
        ############################################################
        #Place your code here



        ############################################################
        #DO NOT DELETE THIS LINE
        super().calc_m_d(data)


    
    def calc_V_d(self, data: Data) -> None:
        """
        """
        ############################################################
        #Place your code here



        ############################################################
        #DO NOT DELETE THIS LINE
        super().calc_V_d(data)