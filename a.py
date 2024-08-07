
import numpy as np 
import pandas as pd 
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report,confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier


df = pd.read_csv('E:\mini project\ds project\Voice.csv')
df.head()
df.shape
df.isna().sum()
df['label'].value_counts()

val= [1584,1584]
label = ['male','female']
plt.figure(figsize=(6,8))
plt.pie(val,labels=label)
plt.legend()
plt.show()

corr = df.corr()
sns.set(font_scale=1.5)
plt.figure(figsize=(20,20))
sns.heatmap(corr,annot = True,cmap='coolwarm')
plt.show()

corr = df.corr()
corr = corr[corr>0.85]
plt.figure(figsize=(15,15))
sns.heatmap(corr,annot = True,cmap='coolwarm')
plt.show()
df = df.drop(['meanfreq','centroid',],axis=1) 
sns.boxplot(x=df.maxdom,y=df.label)
plt.show()
sns.boxplot(x=df.dfrange,y=df.label)
plt.show()
df = df.drop(['dfrange','maxdom'],axis=1)
sns.boxplot(x=df['skew'],y=df.label)
plt.show()
sns.boxplot(x=df['kurt'],y=df.label)
plt.show()
df = df.drop(['kurt','skew'],axis=1)
sns.boxplot(x=df['sd'],y=df.label)
plt.show()
sns.boxplot(x=df['IQR'],y=df.label)
plt.show()
df = df.drop(['IQR'],axis=1)
sns.boxplot(x=df['sfm'],y=df.label)
plt.show()
sns.boxplot(x=df['sp.ent'],y=df.label)
plt.show()
df = df.drop('sp.ent',axis=1)
sns.pairplot(df,kind = 'scatterplot',hue='label')
plt.show()
df = df.drop(['maxfun','modindx','minfun'],axis=1) 
corr = df.corr()
corr = corr[corr>0.85]
plt.figure(figsize=(5,5))
sns.heatmap(corr,annot = True,cmap='coolwarm')
plt.show()

X = df.iloc[:,:-1]
y = df.label
X_train,x_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state = 1)


def eval(y_pred,ytest):
    print("Confusion matrix:\n")
    cm = confusion_matrix(y_pred,ytest)
    sns.heatmap(cm,annot = True,xticklabels=["Female","Male"],yticklabels=["Female","Male"])
    plt.show()
    print("Classification Report\n",classification_report(y_pred,ytest))
    
def score(model):
    print("Training score: ",model.score(X_train,y_train))
    print("Test score: ",model.score(x_test,y_test))
    

DTmodel = DecisionTreeClassifier(min_samples_split = 5,max_depth = 10,random_state = 0)
DTmodel.fit(X_train,y_train)
ypred1 = DTmodel.predict(x_test)
ypred1[:5]
score(DTmodel)
eval(ypred1,y_test)
LRmodel = LogisticRegression(n_jobs=3,max_iter=1000,class_weight=0.001,random_state=0)
LRmodel.fit(X_train,y_train)
ypred2 = LRmodel.predict(x_test)
score(LRmodel)
eval(ypred2,y_test)
SVMmodel = SVC(kernel = 'rbf', C=2.0,random_state=0,degree = 3)
SVMmodel.fit(X_train,y_train)
ypred3 = SVMmodel.predict(x_test)
score(SVMmodel)
eval(ypred3,y_test)
Kmodel = KNeighborsClassifier(n_neighbors = 4,metric ='minkowski',p=1,n_jobs=5,algorithm='ball_tree')
Kmodel.fit(X_train,y_train)  
ypred4 = Kmodel.predict(x_test)
score(Kmodel)
eval(ypred4,y_test)
RFmodel = RandomForestClassifier(n_estimators = 1000,max_depth = 11,n_jobs=5,criterion='gini',warm_start=True,min_samples_split=4,oob_score=True)
RFmodel.fit(X_train,y_train)
ypred5 = RFmodel.predict(x_test) 
score(RFmodel)
eval(ypred5,y_test)   