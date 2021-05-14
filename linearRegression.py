#Task: The student must use both Naive Bayes and Linear Regression to classify a sample dataset, then interpret the results.


import matplotlib.pyplot as py
import seaborn as sb
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

df = pd.read_csv('petrol_consumption.csv')

X = df[['Petrol_tax','Average_income', 'Paved_Highways','Population_Driver_licence(%)']]
Y = df[['Petrol_Consumption']]
X_train, X_test, y_train, y_test = train_test_split(X, Y)

model = LinearRegression()
model.fit(X_train, y_train)

predictions = model.predict(X_test)

sb.distplot(y_test - predictions, axlabel="Test - Prediction")
py.show()

myval1 = np.array([10.00,4399,431,0.5520]).reshape(1, -1)
print(model.predict(myval1))

