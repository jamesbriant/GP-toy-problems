############################################################
# Implementation of the cal_example_zerox.m
# Used to verify the code base works without major errors.
############################################################

import numpy as np

from mcmc.mcmc import MCMC
from mcmc.data import Data
from mcmc.chain import Chain
from mcmc.parameter import Parameter

from mcmc.models.matlab.zerox import Model

#################
##### MODEL #####
#################

model = Model(
    params={
        'theta': Parameter(
            'theta',
            np.array([0.5, 0.5, 0.5]),
        ),
        'beta_eta': Parameter(
            'beta_eta',
            np.array([0.5, 0.5, 0.5, 0.5]),
        ),
        # 'beta_delta': Parameter(
        #     'beta_delta',
        #     np.array([1]),
        # ),
        'lambda_eta': Parameter(
            'lambda_eta',
            np.array([1]),
        ),
        'lambda_delta': Parameter(
            'lambda_delta',
            np.array([10]),
        ),
        'lambda_epsilon': Parameter(
            'lambda_epsilon',
            np.array([1]),
        ),
        'lambda_epsilon_eta': Parameter(
            'lambda_epsilon_eta',
            np.array([300]),
        )
    }
)



################
##### DATA #####
################

DATAFIELD = np.loadtxt('data/cal_example_field.txt')
DATACOMP = np.loadtxt('data/cal_example_comp.txt')

# xf = DATAFIELD[:, 0]
# xf = np.log(xf)
# xf = (xf - np.min(xf))/(np.max(xf) - np.min(xf))
xf = np.zeros((8,1))

# xc = DATACOMP[:, 3]
xc = np.zeros((32, 3))
tc = DATACOMP[:, :3]

yf = DATAFIELD[:, 1]
yc = DATACOMP[:, 4]

#Standardize full response using mean and std of yc
yc_mean = np.mean(yc)
yc_std = np.std(yc)
yc_standardized = (yc - yc_mean)/yc_std
yf_standardized = (yf - yc_mean)/yc_std

tc_normalized = np.zeros_like(tc)
for k in range(tc.shape[1]):
    tc_normalized[:, k] = (tc[:, k] - np.min(tc[:, k]))/(np.max(tc[:, k]) - np.min(tc[:, k]))

data = Data(
    x_c = xc, 
    t   = tc_normalized,
    y   = yc_standardized,
    x_f = xf,
    z   = yf_standardized
)




################
##### MCMC #####
################

proposal_widths = {
    'theta': [0.15, 0.15, 0.15],
    'beta_eta': [0.15, 0.15, 0.15, 0.15],
    'lambda_eta': 1,
    'lambda_epsilon_eta': 100,
    'lambda_delta': 5,
    'lambda_epsilon': 1
}

mcmc = MCMC(
    max_iter = 1000,
    model = model,
    data = data,
    proposal_widths = proposal_widths
)

mcmc.run()



####################
##### ANALYSIS #####
####################

output = mcmc.chain._chain
for item, value in output.items():
    print(item, np.min(value), np.max(value))
