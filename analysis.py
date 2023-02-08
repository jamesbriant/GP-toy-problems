import numpy as np
import matplotlib.pyplot as plt
import mcmc.simulator as sim
import random
import mogp_emulator

from mcmc.testfunctions import arrays_to_arraymesh, arraymesh_to_arrays

random.seed(2022)

###########################################
# 1 - Create emulator of the simulator
###########################################

true_params = [1.5]

n = 40
x_locations = np.linspace(0, 3, n)
y_locations = x_locations
t_month = 6
t_hour = 12

variable_params = arrays_to_arraymesh(x_locations, y_locations, t_month, t_hour)
X = variable_params[:, 0]
Y = variable_params[:, 1]

# print(variable_params.shape)

simulator = sim.Simulator(true_params, variable_params)

## Generate a set of simulation data
calibration_params = np.array([1.4]).reshape(-1, 1)
experiment1 = simulator.run(calibration_params)
print(experiment1.shape)
simulation_output = experiment1[:, -1]
regression_params = experiment1[:, [0,1]]

######## Plot
# experiment1_mesh = arraymesh_to_arrays(experiment1, (n, n))

# fig, ax = plt.subplots()
# CS = ax.contourf(x_locations, y_locations, experiment1_mesh)
# # ax.scatter(X, Y, observations)
# cbar = fig.colorbar(CS)
# plt.show()

######## Fit GP
gp = mogp_emulator.GaussianProcess(regression_params, simulation_output, nugget="fit")
gp = mogp_emulator.fit_GP_MAP(gp, n_tries=1)



