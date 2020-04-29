import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

########################################################################
# Multi-variable Linear Regression
########################################################################


# import data2 (house size, number of bedrooms, sale price)
from ex1.SVLR import gradient_descent, predict

try:
    data2 = pd.read_csv("ex1data2.txt", header=None)
except:
    raise FileExistsError
else:
    print("Load data successfully!\n")

# Gimps the data2
########################################################################
print(data2.head())
print(data2.describe())

# plot raw data
fig, axes = plt.subplots(figsize=(12, 4), nrows=1, ncols=2)

axes[0].scatter(data2[0], data2[2], color="b")
axes[0].set_xlabel("Size (Square Feet)")
axes[0].set_ylabel("Prices")
axes[0].set_title("House prices against size of house")
axes[1].scatter(data2[1], data2[2], color="r")
axes[1].set_xlabel("Number of bedroom")
axes[1].set_ylabel("Prices")
axes[1].set_xticks(np.arange(1, 6, step=1))
axes[1].set_title("House prices against number of bedroom")

# Enhance layout
plt.tight_layout()


# Feature Scaling (Normalization)
########################################################################
def featureNormalization(X):
    """
    Take in numpy array of X values and return normalize X values,
    the mean and standard deviation of each feature
    """
    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0)

    X_norm = (X - mean) / std

    return X_norm, mean, std


# feature normalization / variables initialization
data2_n = data2.values
m2 = len(data2_n[:, -1])
X2 = data2_n[:, 0: 2].reshape(m2, 2)
X2, mean_X2, std_X2 = featureNormalization(X2)
X2 = np.append(np.ones((m2, 1)), X2, axis=1)
y2 = data2_n[:, -1].reshape(m2, 1)
theta2 = np.zeros((3, 1))

theta2, J2_history = gradient_descent(X2, y2, theta2, 0.1, 400)
print("h(x) = " + str(round(theta2[0, 0], 2)) + " + " + str(round(theta2[1, 0], 2)) + " x1 + " + str(
    round(theta2[2, 0], 2)) + " x2")

# results
########################################################################
fig2 = plt.figure()
plt.plot(J2_history)
plt.xlabel("Iteration")
plt.ylabel("$J(\Theta)$")
plt.title("Cost function using Gradient Descent")

plt.show()

# feature normalisation of x values
x_sample = featureNormalization(np.array([1650, 3]))[0]
x_sample = np.append(np.ones(1), x_sample)
predict3 = predict(x_sample, theta2)
print("For size of house = 1650, Number of bedroom = 3, we predict a house value of $" + str(round(predict3, 0)))
