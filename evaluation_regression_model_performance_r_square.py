# -*- coding: utf-8 -*-
"""
Created on Sun Jun 14 23:31:40 2020

@author: Dogukan
"""



import pandas as pd
import matplotlib.pyplot as plt
import  numpy as np

df = pd.read_csv("random_forest_regression.csv",sep=";",header=None)

x= df.iloc[:,0].values.reshape(-1,1)
y= df.iloc[:,1].values.reshape(-1,1)

# %% 
from sklearn.ensemble import RandomForestRegressor
rf=RandomForestRegressor(n_estimators=100,random_state=42)

rf.fit(x,y)

y_head=rf.predict(x)

# %% r_square method

from sklearn.metrics import r2_score
print("r_score:",r2_score(y,y_head))

# %% lineer regression da kullanım
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

y_head = linear_reg.predict(x) # maas
plt.plot(x,y_head,color="red")
 # %% 
from sklearn.metrics import r2_score
print("r_score:",r2_score(y,y_head))

















