# Meshailen Chetty
# 29/07/2020
# Task 23
# =================  Polynomial Regression ===================
import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

# Training set
X_train = [[34], [18], [32], [41], [51], [39], [8], [41]] #bonus balls
y_train = [[3], [6], [10], [13], [17], [20], [24], [27]] #june lotto dates

# Testing set
X_test = [[32], [51], [43], [20], [4], [30], [52], [20]] #bonus balls july
y_test = [[1], [4], [8], [11], [15], [18], [22], [25]] #july lotto dates 
# Train the Linear Regression model and plot a prediction
regressor = LinearRegression()
regressor.fit(X_train, y_train)
xx = np.linspace(0, 60, 100)
yy = regressor.predict(xx.reshape(xx.shape[0], 1))
plt.plot(xx, yy)

# Set the degree of the Polynomial Regression model
quadratic_featurizer = PolynomialFeatures(degree=2)

# This preprocessor transforms an input data matrix into a new data matrix of a given degree
X_train_quadratic = quadratic_featurizer.fit_transform(X_train)
X_test_quadratic = quadratic_featurizer.transform(X_test)

# Train and test the regressor_quadratic model
regressor_quadratic = LinearRegression()
regressor_quadratic.fit(X_train_quadratic, y_train)
xx_quadratic = quadratic_featurizer.transform(xx.reshape(xx.shape[0], 1))

# Plot the graph
# plotting the graph

# plotting the prediction line 
plt.plot(xx, regressor_quadratic.predict(xx_quadratic), c='r', linestyle='-.')

# scattering the test data in green
plt.scatter(X_test,y_test,c='pink')
plt.title('Lotto Bonus Ball Prediction -June 2020')
plt.xlabel('Bonus ball numbers')
plt.ylabel('Days of the month')
plt.axis([0, 60, 0, 31])
plt.grid(True)
plt.scatter(X_train, y_train, c='g')

# linear regression
plt.plot(X_train, regressor.predict(X_train),c= 'b')
plt.show()


print(X_train)
print(X_train_quadratic)
print(X_test)
print(X_test_quadratic)

# If you execute the code, you will see that the simple linear regression model is plotted with
# a solid line. The quadratic regression model is plotted with a dashed line and evidently
# the quadratic regression model fits the training data better.
