import numpy as np
import matplotlib.pyplot as plt
import src.simulator as sim
import random
import mogp_emulator

from src.testfunctions import arrays_to_arraymesh, arraymesh_to_arrays

random.seed(2022)

###########################################
# 0 - Check the class and methods work well
###########################################

# true_params = [1.5]

# n = 80
# x_locations = np.linspace(0, 3, n)
# y_locations = x_locations
# t_month = 6
# t_hour = 12

# variable_params = arrays_to_arraymesh(x_locations, y_locations, t_month, t_hour)
# X = variable_params[:, 0]
# Y = variable_params[:, 1]

# simulator = sim.Simulator(true_params, variable_params)

# ## Generate the observation data
# observations = simulator.get_observations()[:,-1]

# ## Generate a set of simulation data
# calibration_params = np.array([1.4]).reshape(-1, 1)
# experiment1 = simulator.run(calibration_params)[:,-1]
# # print(experiment1)

# ######## Plot
# experiment1_mesh = arraymesh_to_arrays(experiment1, (n, n))

# fig, ax = plt.subplots()
# ax.contourf(x_locations, y_locations, experiment1_mesh)
# # ax.scatter(X, Y, observations)
# plt.show()

########################################################
# 1 - Choose 3 locations - fit a GP over the calibration param!
########################################################

# 3 location
x_locations = np.array([0.8, 1.2, 2.4])
y_locations = np.array([0.5, 2.5, 1.9])
t_month = 6
t_hour = 12

variable_params = np.vstack((x_locations, y_locations, [t_month]*3, [t_hour]*3)).T

true_params = [1.5]
param_vals = [0.8, 1.1, 2.2]
calibration_params = np.array(param_vals).reshape(-1, 1)

simulator = sim.Simulator(true_params, variable_params)

## Generate the observation data
observations = simulator.get_observations(variable_params)[:, -1]

## Generate a set of simulation data
experiment1 = simulator.run(calibration_params)
simulation_output = experiment1[:, -1]
regression_params = experiment1[:, [0,1,4]]

## Fit GP
gp = mogp_emulator.GaussianProcess(regression_params, simulation_output)
gp = mogp_emulator.fit_GP_MAP(gp)

print("Correlation lengths = {}".format(gp.theta.corr))
print("Sigma = {}".format(np.sqrt(gp.theta.cov)))
# print("Nugget = {}".format(np.sqrt(gp.theta.nugget)))


# Setting nugget="pivot" (forcing sigma_e=0 noise) drastically changes the other hyperparameter values
# from default (nugget="adaptive") where sigma_e=~0.0006. This should be investigated.

n_plot_points = 80
k_plot = np.linspace(0, 3, n_plot_points)
prediction_variables = np.vstack(([x_locations[0]]*n_plot_points, [y_locations[0]]*n_plot_points, k_plot)).T
prediction_mu, prediction_var, _ = gp.predict(prediction_variables, include_nugget=False)
# print(prediction_mu)
# print(prediction_var)

plt.figure()
plt.fill_between(k_plot,
                    prediction_mu - 1.96*np.sqrt(prediction_var),
                    prediction_mu + 1.96*np.sqrt(prediction_var),
                    alpha=0.5)
plt.plot(k_plot, prediction_mu)
plt.show()





########################################################
# 2 - Choose 3 locations, assume independence spatially - fit a GP over the model params!
########################################################

# # 3 locations
# locations = np.array([1.5, 4, 5.3146])

# # true_params = [1.5]
# param_vals = np.arange(1.1, 1.95, 0.05)

# simulator = sim.Simulator(locations, include_bias=False)

# ## Generate the observation data
# observations = simulator.get_observations()[:, 1:]

# ## Generate a set of simulation data
# experiment1 = simulator.run(param_vals)[:, 1:]
# print(experiment1[0, :])

# plt.figure()
# for i, location in enumerate(locations):
#     plt.scatter([location]*len(param_vals), experiment1[i, :], marker='+')
# plt.scatter(locations, observations, cmap='r')
# plt.xlabel('x')
# plt.ylabel('simulator output/observation')
# plt.show()