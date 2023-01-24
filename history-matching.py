import numpy as np
import matplotlib.pyplot as plt
import src.simulator as sim
import random
import mogp_emulator
# from sklearn.gaussian_process.kernels import Matern
# from sklearn.gaussian_process import kernels as sk_kern
# from sklearn.gaussian_process import GaussianProcessRegressor

from src.testfunctions import arrays_to_arraymesh, arraymesh_to_arrays

plt.rcParams['figure.figsize'] = [15, 8]
random.seed(2022)

true_params = [1.5745]

hm_threshold = 3
observation_var = 5
discrepancy_var = 0

a = 0
b = 3
n = 25
x_locations = np.linspace(a, b, n)
y_locations = x_locations
t_month = 6
t_hour = 12

variable_params = arrays_to_arraymesh(x_locations, y_locations, t_month, t_hour)
X = variable_params[:, 0]
Y = variable_params[:, 1]

simulator = sim.Simulator(true_params, variable_params, include_bias=False)

# create a simulator and evaluate the function at each location for each calibration param.
# Build a GP emulator for each xy location.
# Use these GPs to perform history matching at each xy location.

hm_xy = np.array([
    [0.7, 2.7, t_month, t_hour],
    [1.2, 0.9, t_month, t_hour],
    [2.3, 1.6, t_month, t_hour]
])
# hm_xy = np.array([
#     [0.7, 2.7, t_month, t_hour],
#     [1.2, 0.9, t_month, t_hour],
#     [0.2, 0.4, t_month, t_hour],
#     [2.7, 1.5, t_month, t_hour],
#     [1.8, 0.1, t_month, t_hour],
#     [2.3, 1.6, t_month, t_hour]
# ])
hm_xy_n = hm_xy.shape[0]


## Generate a set of simulation data
# calibration_points = [1.01, 1.1, 1.15, 1.2, 1.3, 1.35, 1.5, 1.62, 1.7, 1.8, 1.9, 1.99]
# calibration_points = [1.2, 1.5, 1.9]
# calibration_params = np.array(calibration_points).reshape(-1, 1)
# experiment1 = simulator.run(calibration_params)
# print(experiment1.shape)
# simulation_output = experiment1[:, -1]
# regression_params = experiment1[:, [0,1,4]]

calibration_points = [0.1, 0.4, 0.7, 1, 1.3, 1.6, 1.9, 2.2, 2.5, 2.8, 3.1, 3.4, 3.7]
# calibration_points = [0.2, 0.4, 1.2, 2.2, 3.1, 3.2]
# calibration_points = [0.2, 1.2, 2.2, 3.2]


experiment1 = simulator.run(np.array(calibration_points).reshape(-1, 1), hm_xy)
# print(experiment1.shape)

# theta_plot = np.linspace(1, 2, 2000)
theta_plot = np.linspace(0, 4, 500)
implausibility_2_max = 0

fig, axes = plt.subplots(1, hm_xy_n+1, sharex=True, sharey=True)

for i in range(hm_xy_n):
    # select the correct rows for building the emulators at each location
    location_i = [hm_xy_n*x + i for x in range(len(calibration_points))]


    ### Build emulators
    print("Simulation data:", experiment1[location_i, -1])
    gp = mogp_emulator.GaussianProcess(experiment1[location_i, -2], experiment1[location_i, -1], nugget="fit")
    gp = mogp_emulator.fit_GP_MAP(gp, n_tries=10)


    ### Make predictions and build 
    gp_mean, gp_var, _ = gp.predict(theta_plot)
    observation = simulator.get_observations(experiment1[i, :-2][np.newaxis, :])[0, -1] 

    # for counter in range(20):
    #     print( 
    #         gp_mean[100*counter], 
    #         gp_var[100*counter]
    #     )
    print("Emulator variance min/max:", np.min(gp_var), np.max(gp_var))


    # Plot estimates from the emulators
    # axes[i].fill_between(theta_plot, gp_mean+3*np.sqrt(gp_var), gp_mean-3*np.sqrt(gp_var), color="C0", alpha=0.3)
    # axes[i].plot(theta_plot, gp_mean, color="C0")


    ### HISTORY MATCHING
    print("Observation:", observation)
    distance = (observation - gp_mean)**2
    hm_var = gp_var + observation_var + discrepancy_var

    implausibility_2 = distance/hm_var
    implausibility_2_max = np.maximum(implausibility_2_max, implausibility_2)

    # Plot implausibility
    axes[i].scatter(theta_plot, np.sqrt(implausibility_2), s=2)
    axes[i].hlines(hm_threshold, np.min(theta_plot), np.max(theta_plot), linestyles="dashed")
    axes[i].set_ylim((0, 8))
    axes[i].vlines(true_params[0], 0, 8, color="red", linestyles="dashed")
    axes[i].title.set_text(f'location {i+1}')

axes[-1].scatter(theta_plot, np.sqrt(implausibility_2_max), s=2, c="purple")
axes[-1].hlines(hm_threshold, np.min(theta_plot), np.max(theta_plot), linestyles="dashed")
axes[-1].set_ylim((0, 8))
axes[-1].vlines(true_params[0], 0, 8, color="red", linestyles="dashed")
axes[-1].title.set_text('I_max()')

fig.supxlabel("theta")
fig.supylabel("implausibility")
plt.show()