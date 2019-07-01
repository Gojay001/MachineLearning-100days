#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Step1: Data Preprocessing
import pandas as pd
import numpy as np
## Importing the dataset
dataset = pd.read_csv('50_Startups.csv')
X = dataset.iloc[ : , : -1].values
Y = dataset.iloc[ : , 4].values
## Encoding Categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelEncoder = LabelEncoder()
X[:, 3] = labelEncoder.fit_transform(X[:, 3])
# print(X[:, 3])
onehotencoder = OneHotEncoder(categories = 'auto')
X = onehotencoder.fit_transform(X).toarray()
## Avoiding Dummy Variable Trap
X = X[ : , 1:]
## Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 0)
# print(X_train, '\n', X_test, '\n', Y_train, '\n', Y_test)

# Step2: Fitting Multiple Linear Regression to the Training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, Y_train)

# Step3: Predicting the Test set results
Y_pred = regressor.predict(X_test)
print('Y_test = ', Y_test, '\n', 'Y_pred = ', Y_pred)