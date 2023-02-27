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
        'beta1': Parameter(
            'beta1',
            np.array([0.1]),
        ),
        'beta2': Parameter(
            'beta2',
            np.array([0]),
        ),
        'theta': Parameter(
            'theta',
            np.array([1]),
        ),
        'rho': Parameter(
            'rho',
            np.array([1]),
            positive=True,
        ),
        'l_c1_x': Parameter(
            'l_c1_x',
            np.array([0.25, 0.25]),
            positive=True,
        ),
        'l_c1_t': Parameter(
            'l_c1_x',
            np.array([0.25]),
            positive=True,
        ),
        'sigma_c1': Parameter(
            'l_c1_x',
            np.array([1]),
            positive=True,
        ),
        'l_c2_x': Parameter(
            'l_c1_x',
            np.array([0.1, 0.1]),
            positive=True,
        ),
        'sigma_c2': Parameter(
            'l_c1_x',
            np.array([1]),
            positive=True,
        ),
        'lambda': Parameter(
            'l_c1_x',
            np.array([1]),
            positive=True,
        )
    }
)



################
##### DATA #####
################

true_params = [1.5745]

a = 0
b = 3
n = 10
x_locations = np.linspace(a, b, n)
y_locations = x_locations
t_month = 6
t_hour = 12

variable_params = arrays_to_arraymesh(x_locations, y_locations, t_month, t_hour)
X = variable_params[:, 0]
Y = variable_params[:, 1]

simulator = Simulator(true_params, variable_params)

## Generate a set of simulation data
# calibration_points = [1.1, 1.15, 1.2, 1.5, 1.7, 1.8, 1.9]
calibration_points = [1.2, 1.5, 1.9]
calibration_params = np.array(calibration_points).reshape(-1, 1)
experiment1 = simulator.run(calibration_params)
print(experiment1.shape)
simulation_output = experiment1[:, -1]
regression_params = experiment1[:, [0,1,4]]

N = 30
rng = np.random.default_rng()
x_obs = a+(b-a)*rng.random(N)
y_obs = a+(b-a)*rng.random(N)
observation_variables = np.vstack([x_obs, y_obs, [t_month]*N, [t_hour]*N]).T

observations = simulator.get_observations(observation_variables)[:, -1]

xc  = regression_params[:, [0,1]]
t   = regression_params[:, 2].reshape(-1, 1)
y   = simulation_output
xf  = observation_variables[:, [0,1]]
z   = observations

y_mean = np.mean(y)
y_std = np.std(y)
y_normalised = ((y - y_mean)/y_std)

z_mean = np.mean(z)
z_std = np.std(z)
z_normalised = ((z - z_mean)/z_std)

xc_standardised = np.zeros_like(xc)
xc_min = np.min(xc, axis=0).flatten()
xc_max = np.max(xc, axis=0).flatten()
xf_standardised = np.zeros_like(xf)
xf_min = np.min(xf, axis=0).flatten()
xf_max = np.max(xf, axis=0).flatten()

for k in range(2):
    xc_standardised[:, k] = (xc[:, k] - xc_min[k])/(xc_max[k] - xc_min[k])
    xf_standardised[:, k] = (xf[:, k] - xf_min[k])/(xf_max[k] - xf_min[k])

t_min = np.min(t)
t_max = np.max(t)
t_standardised = (t - t_min)/(t_max - t_min)

data = Data(
    x_c = xc_standardised, 
    t   = t_standardised,
    y   = y_normalised,
    x_f = xf_standardised,
    z   = z_normalised
)




################
##### MCMC #####
################

proposal_widths = {
    'beta1': 0.2,
    'beta2': 0.2,
    'theta': 0.3,
    'rho': 0.2,
    'l_c1_x': [0.1, 0.1],
    'l_c1_t': 0.1,
    'l_c2_x': [0.4, 0.4],
    'sigma_c1': 0.15,
    'sigma_c2': 0.2,
    'lambda': 0.5
}

mcmc = MCMC(
    max_iter = 250,
    model = model,
    data = data,
    proposal_widths = proposal_widths
)

mcmc.run()


output = mcmc.chain._chain
for item, value in output.items():
    print(item, np.min(value), np.max(value))
