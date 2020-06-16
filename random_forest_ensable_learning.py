# -*- coding: utf-8 -*-
"""
Created on Tue Jun 16 23:32:17 2020

@author: Dogukan
"""

import pandas as pd 
import numpy as np
# %% import data
data = pd.read_csv("original.csv")
data.drop(["id","Unnamed: 32"],axis=1,inplace=True)
#  %% 
data.diagnosis=[1 if i == "M" else 0for i in data.diagnosis]
y=data.diagnosis.values
x_data=data.drop(["diagnosis"],axis=1)
# && Normalization
x=(x_data-np.min(x_data))/(np.max(x_data)-np.min(x_data))

# %%  train test split 
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test =train_test_split(x,y,test_size=0.15,random_state=42)
# %% decision tree
from sklearn.tree import DecisionTreeClassifier
dt = DecisionTreeClassifier()
dt.fit(x_train,y_train)

print(" decision score:",dt.score(x_test,y_test))


# %% random forest
from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(n_estimators=100,random_state=1)
rf.fit(x_train,y_train)
print("random forest alg. resut:",rf.score(x_test,y_test))


