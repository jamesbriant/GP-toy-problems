import numpy as np
from copy import deepcopy

from models.Model import Model

from typing import Type

class MCMC:
    """Metropolis-Hastings implementation for Kennedy & O'Hagan framework"""

    def __init__(self, iterations, model0: Type[Model]):
        self.proposal_widths = {
            'beta': 0.2,
            'theta': 0.3,
            'rho': 0.3,
            'l_c1_x': 0.1,
            'l_c1_t': 0.1,
            'l_c2_x': 0.4,
            'sigma_c1': 0.15,
            'sigma_c2': 0.2,
            'lambda': 0.5
        }
        self.params_n = 12 # total number of parameters to be estimated by MCMC

        self.iterations = iterations
        self.model0 = model0

    
    def run(self):
        self.samples = np.zeros((self.iterations, self.params_n))
        rng = np.random.default_rng()
        for iter in range(self.iterations):
            if iter % 10 == 0:
                print(iter)
            # print(iter)

            # beta1
            old_param = self.model0.params['beta1']
            new_param = old_param + (rng.random(1) - 0.5)*self.proposal_widths['beta']
            model1 = deepcopy(self.model0)
            model1.rank_errors = 0
            model1.params['beta1'] = new_param

            model1.prior_beta_1()
            model1.m_d_eval()

            if np.log(rng.random(1)) < model1.get_logpost() - self.model0.get_logpost():
                del self.model0
                self.model0 = model1


            # beta2
            old_param = self.model0.params['beta2']
            new_param = old_param + (rng.random(1) - 0.5)*self.proposal_widths['beta']
            model1 = deepcopy(self.model0)
            model1.params['beta2'] = new_param

            model1.prior_beta_2()
            model1.m_d_eval()

            if np.log(rng.random(1)) < model1.get_logpost() - self.model0.get_logpost():
                del self.model0
                self.model0 = model1


            # theta
            old_param = self.model0.params['theta']
            new_param = old_param + (rng.random(1) - 0.5)*self.proposal_widths['theta']
            model1 = deepcopy(self.model0)
            model1.params['theta'] = new_param

            model1.prior_theta()
            model1.C1D1D2_eval()
            model1.V_d_eval()

            if np.log(rng.random(1)) < model1.get_logpost() - self.model0.get_logpost():
                del self.model0
                self.model0 = model1


            # rho
            old_param = self.model0.params['rho']
            new_param = old_param + (rng.random(1) - 0.5)*self.proposal_widths['rho']
            if new_param > 0:
                model1 = deepcopy(self.model0)
                model1.params['rho'] = new_param

                model1.prior_rho()
                model1.H_eval()
                model1.m_d_eval()
                model1.V_d_eval()

                if np.log(rng.random(1)) < model1.get_logpost() - self.model0.get_logpost():
                    del self.model0
                    self.model0 = model1


            # l_c1_x
            for j in range(2):
                old_param = self.model0.hyperparams['l_c1_x'][j]
                new_param = old_param + (rng.random(1) - 0.5)*self.proposal_widths['l_c1_x']
                if new_param > 0:
                    model1 = deepcopy(self.model0)
                    model1.hyperparams['l_c1_x'][j] = new_param

                    model1.prior_l_c1_x()
                    model1.V1D1_eval()
                    model1.V1D2_eval()
                    model1.C1D1D2_eval()
                    model1.V_d_eval()

                    if np.log(rng.random(1)) < model1.get_logpost() - self.model0.get_logpost():
                        del self.model0
                        self.model0 = model1


            # l_c1_t
            old_param = self.model0.hyperparams['l_c1_t']
            new_param = old_param + (rng.random(1) - 0.5)*self.proposal_widths['l_c1_t']
            if new_param > 0:
                model1 = deepcopy(self.model0)
                model1.hyperparams['l_c1_t'] = new_param

                model1.prior_l_c1_t()
                model1.V1D1_eval()
                model1.V1D2_eval()
                model1.C1D1D2_eval()
                model1.V_d_eval()

                if np.log(rng.random(1)) < model1.get_logpost() - self.model0.get_logpost():
                    del self.model0
                    self.model0 = model1


            # l_c2_x
            for j in range(2):
                old_param = self.model0.hyperparams['l_c2_x'][j]
                new_param = old_param + (rng.random(1) - 0.5)*self.proposal_widths['l_c2_x']
                if new_param > 0:
                    model1 = deepcopy(self.model0)
                    model1.hyperparams['l_c2_x'][j] = new_param

                    model1.prior_l_c2_x()
                    model1.V2D2_eval()
                    model1.V_d_eval()

                    if np.log(rng.random(1)) < model1.get_logpost() - self.model0.get_logpost():
                        del self.model0
                        self.model0 = model1


            # sigma_c1
            old_param = self.model0.hyperparams['sigma_c1']
            new_param = old_param + (rng.random(1) - 0.5)*self.proposal_widths['sigma_c1']
            if new_param > 0:
                model1 = deepcopy(self.model0)
                model1.hyperparams['sigma_c1'] = new_param

                model1.prior_sigma_c1()
                model1.V1D1_scale()
                model1.V1D2_scale()
                model1.C1D1D2_scale()
                model1.V_d_eval()

                if np.log(rng.random(1)) < model1.get_logpost() - self.model0.get_logpost():
                    del self.model0
                    self.model0 = model1


            # sigma_c2
            old_param = self.model0.hyperparams['sigma_c2']
            new_param = old_param + (rng.random(1) - 0.5)*self.proposal_widths['sigma_c2']
            if new_param > 0:
                model1 = deepcopy(self.model0)
                model1.hyperparams['sigma_c2'] = new_param

                model1.prior_sigma_c2()
                model1.V2D2_scale()
                model1.V_d_eval()

                if np.log(rng.random(1)) < model1.get_logpost() - self.model0.get_logpost():
                    del self.model0
                    self.model0 = model1


            # lambda
            old_param = self.model0.hyperparams['lambda']
            new_param = old_param + (rng.random(1) - 0.5)*self.proposal_widths['lambda']
            if new_param > 0:
                model1 = deepcopy(self.model0)
                model1.hyperparams['lambda'] = new_param

                model1.V_d_eval()

                if np.log(rng.random(1)) < model1.get_logpost() - self.model0.get_logpost():
                    del self.model0
                    self.model0 = model1


            # print(iter, model1.rank_errors)
            # save values
            self.samples[iter, :] = self.model0.get_model_params()['values']


    # def propose_beta(self):
    #     pass


    # def propose_theta(self):
    #     pass


    # def propose_rho(self):
    #     pass


    # def propose_l_c1_x(self):
    #     pass


    # def propose_l_c1_t(self):
    #     pass


    # def propose_l_c2_x(self):
    #     pass


    # def propose_sigma_c1(self):
    #     pass


    # def propose_sigma_c2(self):
    #     pass


    # def propose_lambda(self):
    #     pass