# -*- coding: utf-8 -*-
"""
Created on Fri Jul 29 16:09:47 2022

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
from sklearn.svm import SVC
model=SVC(kernel='linear',random_state=0)
model.fit(x_train,y_train)
y_pred=model.predict(x_test)
from sklearn.metrics import confusion_matrix,accuracy_score,classification_report
cm=confusion_matrix(y_test, y_pred)
cr=classification_report(y_test, y_pred)
acc=accuracy_score(y_test, y_pred)