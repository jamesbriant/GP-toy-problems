#2022-08-01
##############################################
#History Matching code for James.
#This code is for 1D toy model.
#This code requires two input files,
#"Observation.csv" and "Simulation.csv".
##############################################
#
#
#
#
#
#
#
##############################################
#1. Preparation
##############################################
#
#
#Import modules
import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.gaussian_process.kernels import Matern
from sklearn.gaussian_process import kernels as sk_kern
from sklearn.gaussian_process import GaussianProcessRegressor
#
#
#Import observation data
df_obs = pd.read_csv("Observation.csv",index_col = 0)
observation_data = [df_obs["observed_value"][0]]
obs_error_data = [df_obs["observation_error"][0]]
#
#
#Import simulation data
df_sim = pd.read_csv("Simulation.csv",index_col = 0)
x_training = np.array(df_sim["x_val"])
y_training = np.array(df_sim["y_val"])
#
#
#Setting the threshold.
threshold1 = 3
#
#
#Setting discrepancy variance
error_discrep= 0
#
#
#Plotting observation and simulation outputs
xmin1 = -5
xmax1 = 55
ymin1 = -2
ymax1 = 2
plt.scatter(x_training, y_training,label="Simulation output")
plt.hlines(observation_data, xmin1, xmax1, "blue", linestyles='dashed',label="Observation")
plt.legend()
plt.xlim([xmin1,xmax1])
plt.ylim([ymin1,ymax1])
#plt.savefig("simulation_observation.png")
plt.show()
##############################################
#2. Function settings
##############################################
#
#
#Variance computation.
def Variance_func(X, error_emulator, error_obs, error_discrep):
    Vs = np.zeros(len(X))
    Vs = Vs + error_emulator
    Vs = Vs + error_obs # variance on expectation
    Vs = Vs + error_discrep # model discrepancy
    Vs = np.sqrt(Vs)
    return Vs
#
#
#Plotting implausibility
def plot_plausible(X_, Implausibility_, threshold_):
    m1 = pd.DataFrame(X_[:, 0])
    m2 = pd.DataFrame(Implausibility_)
    m3 = pd.concat([m1, m2], axis=1)
    m3.columns=["X", "implausibility"]
    m4 = m3.copy()
    list1 = []
    for i in range(len(m3)):
        if m3.iloc[i,1] > threshold_:
            m4.iloc[i,1] = None
            list1.append(0)
        else:
            list1.append(1)
    m4["check_num"] = list1
    return m4
#
#
#Simulation model
def y_simulated_1D(x):
    n_points = len(x)
    f = np.zeros(n_points)
    for i in range(n_points):
        f[i] = np.sin(2.*np.pi*x[i] / 50.)
    return f
#
#
#
##############################################
#3. Gaussina Process regression
##############################################
#
#
#Setting Matern Kernel
kernel1 = sk_kern.ConstantKernel(0.1, (1e-3, 1e3)) *Matern(length_scale=1,length_scale_bounds=(1e-05, 100000.0), nu=2.5)+sk_kern.WhiteKernel()
#
#
#Setting parameters and so on for the regression
reg1 = GaussianProcessRegressor(
            kernel=kernel1,
            alpha=1e-10,
            optimizer="fmin_l_bfgs_b",
            n_restarts_optimizer=100,
            normalize_y=True
           )
#
#
#GP fitting requires matrix format, not array format
#So, it is necessary to change the format.
input1 = x_training.reshape(-1, 1)
#
#
#Fitting the simulation data using Gaussian Process
reg1.fit(input1, y_training)
#
#
#For the prediction by Gaussian Proces regression,
#setting the prediction range.
n_rand = 2000
x_predict_min = xmin1
x_predict_max = xmax1
#
#
#Set the random seed.
np.random.seed(0)
#
#
#predictions are at 2,000 points somewhere
#between xmin1 and xmax1.
x_predict = np.random.rand(n_rand)
x_predict = np.sort(x_predict, axis=0)
x_predict *= (x_predict_max - x_predict_min)
x_predict += x_predict_min
#
#
#Again, changing the format.
x_predict = x_predict[:,None]
#
#
#
#Prediction by Gaussian Process regression.
y_pred, MSE = reg1.predict(x_predict, return_std=True, )
#
#
#Plotting the result.
plt.plot(x_predict[:, 0], y_pred, color="C0", label="predict mean")
plt.fill_between(x_predict[:, 0], y_pred + 3*MSE, y_pred - 3*MSE, color="C0", alpha=.3,label= "3 sigma confidence")
plt.plot(x_training, y_training, "o",label= "Simulation output")
plt.hlines(observation_data, xmin1, xmax1, "blue", linestyles='dashed',label="Observation")
#plt.title("GP by Scikit-learn")
plt.legend()
#plt.savefig("GP_fitting_by_Matern.png")
plt.show()
#
#
#
##############################################
#4. History Matching
##############################################
#
#
#Computing the distance between the observation
#and expectation fo emulated outputs.
distance = np.abs(observation_data[0]-y_pred)
#
#
#Computing variance.
Vs1 = Variance_func(x_predict[:, 0], MSE*MSE, obs_error_data, error_discrep)
#
#
#Computing implausibility.
Implausibility1 = (distance / Vs1)
#
#
#
ymin2 = -0.5
ymax2 = 20
Implausibility2 = plot_plausible(x_predict, Implausibility1, threshold1)
plt.scatter(x_predict[:, 0], Implausibility1, s = 2, c = "black")
plt.scatter(Implausibility2.iloc[:,0],Implausibility2.iloc[:,1], s = 2, c = "green")
plt.ylim([ymin2,ymax2])
plt.hlines([threshold1], min(x_predict[:, 0]), max(x_predict[:, 0]), "blue", linestyles='dashed')
#plt.savefig("matern_implausibility.png")
plt.show()
#
#
#
##############################################
#5. History Matching Second wave start
##############################################
#
#
#Picking up plausible subspace elements
list2 = []
for i in range(len(Implausibility2)):
    if Implausibility2["check_num"][i] == 1:
        list2.append(Implausibility2["X"][i])
#
#
#Chosing an additional
random.seed(100)
num1 = random.randint(1, len(list2))
#
#
#Adding the additional point to the traning data
x_list = list(x_training)
x_list.append(list2[num1])
x_training = np.array(x_list)
y_training = y_simulated_1D(x_training)
#
#
#Plotting the simulation output
plt.scatter(x_training, y_training,label="Simulation output")
plt.hlines(observation_data, xmin1, xmax1, "blue", linestyles='dashed',label="Observation")
plt.legend()
plt.xlim([xmin1,xmax1])
plt.ylim([ymin1,ymax1])
#plt.savefig("Additional_simulation_observation.png")
plt.show()
#
#
#
##############################################
#6. Gaussian Process regression.
##############################################
#
#
#Setting Matern Kernel
kernel1 = sk_kern.ConstantKernel(0.1, (1e-3, 1e3)) *Matern(length_scale=1,length_scale_bounds=(1e-05, 100000.0), nu=2.5)+sk_kern.WhiteKernel()
#
#
#Setting parameters and so on for the regression
reg1 = GaussianProcessRegressor(
            kernel=kernel1,
            alpha=1e-10,
            optimizer="fmin_l_bfgs_b",
            n_restarts_optimizer=100,
            normalize_y=True
           )
#
#
#GP fitting requires matrix format, not array format
#So, it is necessary to change the format.
input1 = x_training.reshape(-1, 1)
#
#
#Fitting the simulation data using Gaussian Process
reg1.fit(input1, y_training)
#
#
#For the prediction by Gaussian Proces regression,
#setting the prediction range.
n_rand = 2000
x_predict_min = xmin1
x_predict_max = xmax1
#
#
#Set the random seed.
np.random.seed(0)
#
#
#predictions are at 2,000 points somewhere
#between xmin1 and xmax1.
x_predict = np.random.rand(n_rand)
x_predict = np.sort(x_predict, axis=0)
x_predict *= (x_predict_max - x_predict_min)
x_predict += x_predict_min
x_predict = x_predict[:,None]
#
#
#Prediction by Gaussian Process regression.
y_pred, MSE = reg1.predict(x_predict, return_std=True, )
#
#
#Plotting the result
plt.plot(x_predict[:, 0], y_pred, color="C0", label="predict mean")
plt.fill_between(x_predict[:, 0], y_pred + 3*MSE, y_pred - 3*MSE, color="C0", alpha=.3,label= "3 sigma confidence")
plt.plot(x_training, y_training, "o",label= "simulation data")
plt.plot(x_training[len(x_training)-1], y_training[len(y_training)-1], "o",label= "additional simulation data",color="red")
plt.hlines([observation_data], min(x_predict[:, 0]), max(x_predict[:, 0]), "blue", linestyles='dashed')
#plt.title("GP by Scikit-learn")
plt.legend()
#plt.savefig("matern_fit2.png")
plt.show()
#
#
#
##############################################
#7. History Matching Second wave
##############################################
#
#
#Computing the distance between the observation
#and expectation fo emulated outputs.
distance = np.abs(observation_data[0]-y_pred)
#
#
#Computing variance.
Vs1 = Variance_func(x_predict[:, 0], MSE*MSE, obs_error_data, error_discrep)
#
#
#Computing implausibility.
Implausibility1 = (distance / Vs1)
#
#
#
Implausibility2 = plot_plausible(x_predict, Implausibility1, threshold1)
plt.scatter(x_predict[:, 0], Implausibility1, s = 2, c = "black")
plt.scatter(Implausibility2.iloc[:,0],Implausibility2.iloc[:,1], s = 2, c = "green")
plt.ylim(ymin2,ymax2)
plt.hlines([3], min(x_predict[:, 0]), max(x_predict[:, 0]), "blue", linestyles='dashed')
#plt.savefig("matern_implausibility2.png")
plt.show()
