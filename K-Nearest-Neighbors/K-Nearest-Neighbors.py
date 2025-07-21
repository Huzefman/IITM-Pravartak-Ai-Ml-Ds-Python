#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 21 16:27:49 2025

@author: huzefa
"""

import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix,accuracy_score

#Importing data
data=pd.read_csv('../Datasets/income.csv')

#Getting basic information and description of data
print(data.info())
print(data.describe())
data_desc=data.describe(include='O')

#Checking for unique values in order to obtain any dirty character present in data
uniqueVals={}
for col in data.columns:
    uniqueVals[col]=data[col].unique()
for col,values in uniqueVals.items():
    print()
    print(f"Unique values in '{col}':{values}")
    
#Rereading data after removing ' ?' from data and replacing it with NAN
data1=pd.read_csv('../DataSets/income.csv',na_values=[' ?'])

#Checking again
print(data1.info())
print(data1.describe())
data_desc=data1.describe(include='O')
uniqueVals={}
for col in data1.columns:
    uniqueVals[col]=data1[col].unique()
for col,values in uniqueVals.items():
    print()
    print(f"Unique values in '{col}':{values}")
    
#Checking for total number null values present in JobType and occupation
print(data1['JobType'].isnull().sum())
print(data1['occupation'].isnull().sum())
missing=data1[data1.isnull().any(axis=1)]

#Deleting this rows with missing values as both are related and most of them belong to same rows
#The other remaining 7 values in occupation has JobType as Never worked
data2=data1.dropna(axis=0)
print(data2.info())

#Mapping SalStat values to 0 and 1
data2['SalStat']=data2['SalStat'].map({' less than or equal to 50,000':0, ' greater than 50,000':1})
print(data2['SalStat'])

#Converting categorical values to dummies
newVals=pd.get_dummies(data2, drop_first=True)

#Storing list of columns from newVals
columns_list=list(newVals.columns)
print(columns_list)

#Extracting feature columns as input
features=list(set(columns_list)-set(['SalStat']))
print(features)

#Extracting SalStat values as y
y=newVals['SalStat'].values
print(y)

#Extracting all the feature values
x=newVals[features].values
print(x)

#Splitting data into train and test
train_x,test_x,train_y,test_y=train_test_split(x,y, test_size=0.3,random_state=0)

#Storing K Nearest Classifier
KNN_classifier= KNeighborsClassifier(n_neighbors=5)

#Fitting train_x and train_y values
KNN_classifier.fit(train_x, train_y)

#Predicting test values with model
prediction=KNN_classifier.predict(test_x)

#Performance Matrix Check
confusionMatrix=confusion_matrix(test_y, prediction)
print(confusionMatrix)

#Accuracy Check
accuracy=accuracy_score(test_y,prediction)
print(accuracy)

#Misclassified predictions count
missclassified=('Missclassified values: %d'%(prediction!=test_y).sum())
print(missclassified)

#Getting k values for classifier upto 20 to check for different accuracy levels
missclassified_sample=[]
for i in range(1,20):
    KNN=KNeighborsClassifier(n_neighbors=i)
    KNN.fit(train_x, train_y)
    pred_i=KNN.predict(test_x)
    missclassified_sample.append((pred_i!=test_y).sum())
print(missclassified_sample)

#Taking n_neighbors value as 16 give least misclassified predictions hence more accurate
knn=KNeighborsClassifier(n_neighbors=16)
knn.fit(train_x,train_y)
pred=knn.predict(test_x)
acc=accuracy_score(test_y, pred)
print(acc)
print('Missclassified values taking n=16 are: %d'%(test_y!=pred).sum())