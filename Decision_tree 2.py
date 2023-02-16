# -*- coding: utf-8 -*-
"""
Created on Thu May  6 19:38:15 2021

@author: 91703
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


dataset = pd.read_csv("Social_Network_Ads.csv")

x = dataset.iloc[:,0:2].values
y = dataset.iloc[:,2].values

#Visualization of dataset
plt.scatter(x[y==0,0],x[y==0,1],color="red",label="Not purchased")
plt.scatter(x[y==1,0],x[y==1,1],color="green",label="Purchased")
plt.xlabel("Age")
plt.ylabel("Estimated Salary")
plt.title("Car purchased with age and estimated salary")
plt.legend()
plt.show()

#feature scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x = sc.fit_transform(x)

#Spliting into training and testing
from sklearn.model_selection import train_test_split
X_train,X_test,y_train, y_test = train_test_split(x,y,train_size=0.75,random_state=0)
    
#fitting training set into model
from sklearn.tree import DecisionTreeClassifier
dt_model=DecisionTreeClassifier(criterion="gini")
dt_model.fit(X_train,y_train)


from sklearn.linear_model import LogisticRegression
lg_model = LogisticRegression()
lg_model.fit(X_train,y_train)

#Tree visualisation
from sklearn.tree import export_text 
tree=export_text(dt_model,feature_names=["Age","salary"])

y_pred=dt_model.predict(X_test)
y_pred1=dt_model.predict(X_train)

y_pred9 = lg_model.predict(X_test) 

#confusion matrix
from sklearn.metrics import confusion_matrix,accuracy_score,classification_report
cm=confusion_matrix(y_test,y_pred)
cm1=confusion_matrix(y_train,y_pred1)

cr=classification_report(y_test,y_pred)

#Visualisation of ROC AUC curve
from sklearn.metrics import roc_auc_score,roc_curve,auc
fpr,tpr,thresh=roc_curve(y_test,y_pred)
a = auc(fpr,tpr)


fpr1,tpr1,thresh = roc_curve(y_test,y_pred9)
b = auc(fpr1,tpr1)

plt.plot(fpr,tpr,color="green",label=("AUC value of Decision tree: %0.2f"%(a)))
plt.plot(fpr1,tpr1,color="blue",label=("AUC value of logistic Regression: %0.2f"%(b)))
plt.plot([0,1],[0,1],"--",color="red")
plt.xlabel("False positive rate")
plt.ylabel("True Positive rate")
plt.title("ROC-AUC Curve")
plt.legend(loc="best")
plt.show()



#Visualization of Testing dataset with model

from matplotlib.colors import ListedColormap
X_set, y_set = X_test, y_test


X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))

plt.contourf(X1, X2, dt_model.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.8, cmap = ListedColormap(('red', 'green')))


plt.scatter(X_set[y_set==0,0],X_set[y_set==0,1],color="red",label="Not purchased")
plt.scatter(X_set[y_set==1,0],X_set[y_set==1,1],color="green",label="Purchased")
plt.xlabel("Age")
plt.ylabel("Estimated Salary")
plt.title("Car purchased with age and estimated salary")
plt.legend()
plt.show()

#Visualization of Training dataset with model

from matplotlib.colors import ListedColormap
X_set, y_set = X_train, y_train


X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))

plt.contourf(X1, X2, dt_model.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.8, cmap = ListedColormap(('red', 'green')))


plt.scatter(X_set[y_set==0,0],X_set[y_set==0,1],color="red",label="Not purchased")
plt.scatter(X_set[y_set==1,0],X_set[y_set==1,1],color="green",label="Purchased")
plt.xlabel("Age")
plt.ylabel("Estimated Salary")
plt.title("Car purchased with age and estimated salary")
plt.legend()
plt.show()


a={'age':50,'salary':100000}
data=pd.DataFrame(a,index=[0])
d=sc.transform(data)

b=dt_model.predict(d)
if b==0:
    print("not")
else:
    print("yes")    



#how to save a model
import pickle
f1=open(file="decision_model.pkl",mode='bw')
pickle.dump(dt_model,f1)
f1.close()
