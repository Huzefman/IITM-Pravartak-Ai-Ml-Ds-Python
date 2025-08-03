#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug  3 13:07:20 2025

@author: huzefa
"""
#Importing necessasary libraries
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.svm import LinearSVC, SVC
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import average_precision_score, recall_score, make_scorer

#Importing dataset
diabetes_data=pd.read_excel('../Datasets/data_file.xlsx')
diabetes_data.info()

#Creting a copy of original df to work upon
clean_data=diabetes_data.copy()

#Cleaning clean_data by pruning duplicates if present
clean_data=clean_data.drop_duplicates(keep='first')
clean_data.info()
print(clean_data.head())

#Assigning lables with appropriate numerics
nondia=0
diabetic=1
Ynew=pd.DataFrame(nondia,index=clean_data.index,columns=['Diabetic'])

#Identifying the diabetic status of each record using blood test result of FBS or HBA1C1
Ynew.iloc[list(np.where((clean_data.OGTT1FBS>=126) | (clean_data.HBA1C1>=6.5))[0])]=diabetic

#Concatenating diabetic status with anthroprometric features of dataset
data_df=pd.concat([clean_data.iloc[:,:6],Ynew],axis=1)
print(data_df.head())

#Finding the diabetic and non diabetic patients
diabetic_yes=data_df.iloc[list(np.where(data_df.Diabetic==diabetic)[0])]
diabetic_no=data_df.iloc[list(np.where(data_df.Diabetic==nondia)[0])]

#Getting basic stats about diabetic people
print(diabetic_yes.describe())

#Getting basic stats about non diabetic people
print(diabetic_no.describe())

#Encoding categorical variables to numeric values (To ensure consistency)
le=LabelEncoder()
data_df.Diabetic=le.fit_transform(data_df.Diabetic)
data_df.GENDER=le.fit_transform(data_df.GENDER)

#Splitting data into training and testing set
train_x, test_x, train_y, test_y=train_test_split(data_df.iloc[:,:6], data_df.Diabetic, test_size=0.3,random_state=43)

#Normalizing data using standard scaler
sc=StandardScaler()
train_x=sc.fit_transform(train_x)
test_x=sc.transform(test_x)

#Fetching diabetic and non diabetic records seperately from train set
train_x_diabetic=train_x[list(np.where(train_y==diabetic)[0])]
train_x_nondiabetic=train_x[list(np.where(train_y==nondia)[0])]

#Fetching diabetic and non diabetic records seperately from test set
test_x_diabetic=test_x[list(np.where(test_y==diabetic)[0])]
test_x_nondiabetic=test_x[list(np.where(test_y==nondia)[0])]

#Writing Functions for displaying evaluations
def evaluate(yt,yp):
    cf=confusion_matrix(yt, yp)
    acc=accuracy_score(yt, yp)
    return cf,acc

def display(yt,yp,model):
    cf,acc=evaluate(yt,yp)
    print('Model=',model,'\nConfusion matrix= ',cf,'\nAccuracy score= ',acc)
    
#Performing classification using linear SVM
lsvc=LinearSVC(random_state=0,C=10,max_iter=100000)
lsvc.fit(train_x, train_y)
train_yp=lsvc.predict(train_x)
test_yp=lsvc.predict(test_x)

#Display the result
display(train_y,train_yp,'Linear SVC: Validation')
display(test_y, test_yp,'Linear SVC: Testing')

#Coefficients of each feature in scaled x
print(lsvc.coef_)

#Intercept at scaled x
print(lsvc.intercept_)

#Rescaling the coefficients to original scale of the features of X
rescaled_coef=lsvc.coef_/np.sqrt(sc.var_)
print(rescaled_coef)

#The intercept in the original feature space
rescaled_intercept=rescaled_coef.dot(sc.mean_.T)+lsvc.intercept_
print(rescaled_intercept)

#Now identifying slacks or misclassified points in each class
non_dia_slacks=(lsvc.coef_.dot(train_x_diabetic.T)+lsvc.intercept_)
print(np.sum(non_dia_slacks<0))
dia_slacks=(lsvc.coef_.dot(train_x_nondiabetic.T)+lsvc.intercept_)
print(np.sum(dia_slacks>0))

#Creating custom dictionary for recall and precision
custom_scorer = {'recall':make_scorer(recall_score, pos_label=diabetic),
'precision':make_scorer(average_precision_score, pos_label=diabetic)}

#Tuning regularization parameter and retraining the model using best C value
gscv = GridSearchCV(LinearSVC(max_iter=int(1e7)), {'C':[1e-5,1e-4,1e-3,1e-2,1e-1,1,10,100,1000]},
cv=5,verbose=False,scoring=custom_scorer,refit='recall')
gscv.fit(train_x,train_y)
print(gscv.best_params_)

#Displaying the results based on C value 10 and best found value
display(train_y,train_yp,'For C=10: Training')
lsvc = LinearSVC(random_state=0,C=0.001,max_iter=100000)
lsvc.fit(train_x, train_y)
train_yp=lsvc.predict(train_x)
display(train_y,train_yp,'For C=0.001: Training')
