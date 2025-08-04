#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug  4 16:07:26 2025

@author: huzefa
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, average_precision_score, make_scorer
from sklearn.neural_network import MLPClassifier

#Importing dataset
diabetes=pd.read_excel('../Datasets/data_file.xlsx')
diabetes.info()

#Creating a copy of original data to work upon
clean_data=diabetes.copy()

#Pruning duplicates from copy data
clean_data=clean_data.drop_duplicates(keep='first')
print(clean_data.shape)
print(clean_data.head())

#Assiging labels with appropriate numerics
nondia=-1
diabetic=1

#Creating a dataframe with a column depicting diabetic status
Ynew=pd.DataFrame(nondia,index=clean_data.index,columns=['Diabetic'])

#Identifying the diabetic status of each record using the blood test results of FBS or HBA1C1
Ynew.iloc[list(np.where((clean_data.OGTT1FBS>=126) | (clean_data.HBA1C1>=6.5))[0])]=diabetic

#Concatenating the diabetic status with the anthroprometric features of the dataset
data_df=pd.concat([clean_data.iloc[:,:6],Ynew],axis=1)
print(data_df.head())

#Finding the diabetic and non-diabetic patients
diabetic_yes=data_df.iloc[list(np.where(data_df.Diabetic==diabetic)[0])]
diabetic_no=data_df.iloc[list(np.where(data_df.Diabetic==nondia)[0])]

#Finding basic stats about both classes respectively
print(diabetic_yes.describe())
print(diabetic_no.describe())

#Splitting data into training and testing set
train_x, test_x, train_y, test_y=train_test_split(data_df.iloc[:,:6], data_df.Diabetic,test_size=0.3,random_state=43)

#Normalizing data using standard scalar
sc=StandardScaler()
train_x=sc.fit_transform(train_x)
test_x=sc.transform(test_x)

#Fetching the diabetic records from train set
diabetic_yes_train=train_x[list(np.where(train_y==diabetic)[0])]

#Fetching the non-diabetic records from train set
diabetic_no_train=train_x[list(np.where(train_y==nondia)[0])]

#Displaying the counts for each class
print('non-diabetic=',diabetic_no_train.shape,'diabetic=',diabetic_yes_train.shape)

#Fetching the diabetic records from test set
diabetic_yes_test=test_x[list(np.where(test_y==diabetic)[0])]

#Fetching the non-diabetic records from test set
diabetic_no_test=test_x[list(np.where(test_y==nondia)[0])]

#Displaying the counts for each class from test set
print('non-diabetic=',diabetic_no_test.shape,'diabetic=',diabetic_yes_test.shape)

#Functions for evaluating model using confusion matrix and accuracy score between true and actual
def evaluate(yt,yp):
    cf=confusion_matrix(yt,yp)
    acc=accuracy_score(yt,yp)
    return cf,acc

# Display metrics
def display(yt,yp,model):
    cf,acc = evaluate(yt,yp)
    print('Model=',model,'\ncf=',cf,'\n','\nacc=',acc,'\n')
    
#Performing classification using MLP classifier
mlpc = MLPClassifier(hidden_layer_sizes=(1), activation='tanh', learning_rate='invscaling', max_iter=10000, solver='sgd', random_state=0, early_stopping=True)
mlpc.fit(train_x, train_y)
train_yp=mlpc.predict(train_x)
test_yp=mlpc.predict(test_x)

#Displaying the results
display(train_y,train_yp,'MLP: Training')
display(test_y,test_yp,'MLP: Testing')

#Attributes of the MLP Classifier
print(mlpc.classes_)
print(mlpc.loss_)
print(mlpc.coefs_)
print(mlpc.intercepts_)
print(mlpc.n_layers_)
print(mlpc.n_iter_)
print(mlpc.n_outputs_)
print(mlpc.out_activation_)

#Hyperparameter tuning using GridSearchCV
custom_scorer = {'recall':make_scorer(recall_score, pos_label=diabetic), 'precision':make_scorer(average_precision_score, pos_label=diabetic)}

gscv = GridSearchCV(MLPClassifier(max_iter=10000,random_state=0),
{'activation':('tanh','logistic','relu'),
'hidden_layer_sizes':range(1,4,1),'solver':['adam','sgd']},
cv=5,verbose=False,
scoring=custom_scorer,refit='recall')
gscv.fit(train_x,train_y)
gscv.best_params_

#Performoming Classification using MLP Classifier with best obtained parameters
mlpc = MLPClassifier(hidden_layer_sizes=(1),activation='logistic',
max_iter=10000,
solver='adam',
random_state=0)
mlpc.fit(train_x, train_y)
train_yp=mlpc.predict(train_x)
test_yp=mlpc.predict(test_x)

#Displaying the results
display(train_y,train_yp,'MLP with "sgd" solver and 1,4 hidden nodes')
display(test_y,test_yp,'For Testing')

#Attributes of improved model
print(mlpc.coefs_)
print(mlpc.intercepts_)
print(mlpc.score(test_x,test_y))