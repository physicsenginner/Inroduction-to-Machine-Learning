# -*- coding: utf-8 -*-
"""
Created on Sun Jun 14 22:29:33 2020

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
print("7.5 seviyesinde fiyatin ne kadar oldugu:",rf.predict([[7.5]]))

x_=np.arange(min(x),max(x),0.01).reshape(-1,1)
y_head=rf.predict(x_)

#visualize

plt.scatter(x,y,color="red")
plt.plot(x_,y_head,color="green")
plt.xlabel("tribun level")
plt.ylabel("ucret")
plt.show()
