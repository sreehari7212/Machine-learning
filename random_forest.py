# -*- coding: utf-8 -*-
"""
Created on Mon Jul 25 14:47:18 2022

@author: SREEHARI CR
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
df=pd.read_csv("Social_Network_Ads.csv")
x = df.iloc[:,0:2].values
y = df.iloc[:,2].values
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=.25,random_state=0)
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
x_train=sc.fit_transform(x_train)
x_test=sc.transform(x_test)
from sklearn.ensemble import RandomForestClassifier
model=RandomForestClassifier(n_estimators=5,criterion='entropy')
model.fit(x_train,y_train)
y_pred=model.predict(x_test)
y_pred1=model.predict(x_train)
from sklearn.metrics import confusion_matrix,accuracy_score,classification_report
cm=confusion_matrix(y_test, y_pred)
cm1=confusion_matrix(y_train,y_pred1)
cr=classification_report(y_test, y_pred)
acc=accuracy_score(y_test, y_pred)
