import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

########################################################################
# Single variable Linear Regression
########################################################################


# Import data
try:
    data = pd.read_csv("ex1data1.txt", header=None)
except:
    raise FileExistsError
else:
    print("Load data successfully!\n")

# Gimps the data
########################################################################
print(data.head())
print(data.describe())

# plot raw data
plt.scatter(data[0], data[1])
plt.xticks(np.arange(5, 30, step=5))
plt.yticks(np.arange(-5, 30, step=5))
plt.xlabel("Population of City (10,000s)")
plt.ylabel("Profit ($10,000")
plt.title("Profit Prediction")


# plt.show()


# Cost function
########################################################################
def compute_cost(X, y, theta):
    """
    Take in a numpy array X,y, theta and calculate the cost function
    using theta as parameter in a linear regression model
    """
    m = len(y)
    y_predict = X.dot(theta)
    error = y_predict - y

    return 1 / (2 * m) * np.sum(error ** 2)


# variables initialization
data_n = data.values
m = data_n[:, 0].size
X = np.append(np.ones((m, 1)), data_n[:, 0].reshape(m, 1), axis=1)
y = data_n[:, 1].reshape(m, 1)
theta = np.zeros((2, 1))

cost = compute_cost(X, y, theta)
print('The cost when using zeros as theta is: %f' % cost)


# Gradient Descent
########################################################################
def gradient_descent(X, y, theta, alpha, num_iter):
    """
    Take in numpy array X, y and theta and update theta by
    taking num_iter gradient steps with learning rate of alpha

    return theta and the list of the cost of theta during each iteration
    """
    m = len(y)
    cost_his = []

    for i in range(num_iter):
        predictions = np.dot(X, theta)
        error = np.dot(X.transpose(), (predictions - y))
        descent = alpha / m * error
        theta -= descent
        cost_his.append(compute_cost(X, y, theta))

    return theta, cost_his


# Show results
########################################################################

# test code for gradient descent
theta, J_history = gradient_descent(X, y, theta, 0.01, 1500)
print("h(x) = " + str(round(theta[0, 0], 2)) + " + " + str(round(theta[1, 0], 2)) + " x1")

# Generating values for theta0, theta1 and the resulting cost value
theta0_vals = np.linspace(-10, 10, 100)
theta1_vals = np.linspace(-1, 4, 100)
J_vals = np.zeros((len(theta0_vals), len(theta1_vals)))

for i in range(len(theta0_vals)):
    for j in range(len(theta1_vals)):
        t = np.array([theta0_vals[i], theta1_vals[j]])
        J_vals[i, j] = compute_cost(X, y, t)

# Generating the surface plot
fig1 = plt.figure()
ax = fig1.add_subplot(111, projection='3d')
surf = ax.plot_surface(theta0_vals, theta1_vals, J_vals, cmap="coolwarm")
fig1.colorbar(surf, shrink=0.5, aspect=5)
ax.set_xlabel("$\Theta_0$")
ax.set_ylabel("$\Theta_1$")
ax.set_zlabel("$J(\Theta)$")
plt.title("Cost vs Thetas")
ax.view_init(33, 133)  # rotate for better view

# learning curve of gradient descent
fig2 = plt.figure()
plt.plot(J_history)
plt.xlabel("Iteration")
plt.ylabel("$J(\Theta)$")
plt.title("Cost function using Gradient Descent")

# best fitting line
fig3 = plt.figure()
plt.scatter(data[0], data[1])
x_value = [x for x in range(25)]
y_value = [y * theta[1] + theta[0] for y in x_value]
plt.plot(x_value, y_value, color="r")
plt.xticks(np.arange(5, 30, step=5))
plt.yticks(np.arange(-5, 30, step=5))
plt.xlabel("Population of City (10,000s)")
plt.ylabel("Profit ($10,000")
plt.title("Profit Prediction")


plt.show()

# Predictions
########################################################################
def predict(x, theta):
    """
    Takes in numpy array of x and theta and return the predicted value of y based on theta
    """
    predictions = np.dot(theta.transpose(), x)

    return predictions[0]


prdt1 = predict(np.array([1, 3.5]), theta) * 10000
print("For population = 35,000, we predict a profit of $" + str(round(prdt1, 0)))
prdt2 = predict(np.array([1, 7]), theta) * 10000
print("For population = 70,000, we predict a profit of $" + str(round(prdt2, 0)))
