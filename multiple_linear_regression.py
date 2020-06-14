# -*- coding: utf-8 -*-
"""
Created on Sun Jun 14 19:10:28 2020

@author: Dogukan
"""
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

df = pd.read_csv("multiple_linear_regression.csv",sep=";")   

x = df.iloc[:,[0,2]].values
y=df.maas.values.reshape(-1,1)
# %%
multiple_linear_regression = LinearRegression()
multiple_linear_regression.fit(x,y)
print("b0:",multiple_linear_regression.intercept_)
print("b1,b2:",multiple_linear_regression.coef_)

multiple_linear_regression.predict([[10,35],[5,35]])








