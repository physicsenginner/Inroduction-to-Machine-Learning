# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
# import library
import pandas as pd
import  matplotlib.pyplot as plt
# import data
df = pd.read_csv("linear_regression_dataset.csv",sep=";")
#plot data
plt.scatter(df.deneyim,df.maas)
plt.xlabel("deneyim")
plt.ylabel("maas")
plt.show()
# %% Linear Regression
# sklearn Library
from sklearn.linear_model import LinearRegression
 # Linear regression model
linear_reg =LinearRegression()
x=df.deneyim.values.reshape(-1,1) #numpy cevirdik
y=df.maas.values.reshape(-1,1)

linear_reg.fit(x,y)
# %% prediction
import numpy as np
b0=linear_reg.predict([[0]])
print("b0:",b0)

b0_ =linear_reg.intercept_ # y eksenini kestigi nokta
print("b0_:",b0_)

b1=linear_reg.coef_  # egim slope
print("b1:",b1)

#maas=1663+1144*deneyim

maas_yeni=1663+1144*11
print(maas_yeni)
print(linear_reg.predict([[11]]))
# visualization
array = np.array([0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]).reshape(-1,1) #deneyim
plt.scatter(x,y)
plt.show()
y_head = linear_reg.predict(array) #maas

plt.plot(array,y_head,color = "red")
linear_reg.predict([[100]])






