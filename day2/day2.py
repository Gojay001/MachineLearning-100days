#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Step1: Data Preprocessing
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

dataset = pd.read_csv('studentscores.csv')
X = dataset.iloc[ : , : 1].values
Y = dataset.iloc[ : , 1].values

from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 1/4, random_state = 0)
# print(X_train, '\n', X_test, '\n', Y_train, '\n', Y_test)

# Step2: Fitting Simple Linear Regression Model to the training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor = regressor.fit(X_train, Y_train)

# Step3: Predicting the Result
Y_pred = regressor.predict(X_test)

# Step4: Visualization
## Visualising the Training results
plt.figure()
plt.scatter(X_train, Y_train, color = 'red') 
plt.plot(X_train, regressor.predict(X_train), color = 'blue')

## Visualizing the Test results
plt.figure()
plt.scatter(X_test, Y_test, color = 'red')
plt.plot(X_test, Y_pred, color = 'blue')
plt.show()