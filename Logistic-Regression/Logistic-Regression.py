#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 21 19:12:44 2025

@author: huzefa
"""

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score

#Importing data
data=pd.read_csv('../DataSets/income.csv')

data1=data.copy()

#Getting basic idea about data
print(data1.info())
print(data1.describe())
cate_desc=data1.describe(include='O')

#Checking for unique values
uniqueVals={}
for col in data1.columns:
    uniqueVals[col]=data1[col].unique()
for col,values in uniqueVals.items():
    print()
    print(f"Unique values in '{col}':{values}")
    
#Replacing ? values from JobType and occupation columns with NAN
data2=pd.read_csv('../DataSets/income.csv',na_values=' ?')
print(data2.info())
data3=data2.copy()

#Checking for columns where atleast one field has NAN
missing = data2[data2.isnull().any(axis=1)]

#Most of Values of NAN are related to other column missing value hence dropping the missing values
data2=data2.dropna(axis=0)
data2.info()

#Getting a gender ratio
gender = pd.crosstab(index = data2["gender"], columns  = 'count', normalize = True)
print(gender)

#Gender vs Salary ratio
gender_salstat = pd.crosstab(index = data2["gender"],columns = data2['SalStat'], margins = True, normalize =  'index')
print(gender_salstat)

#Getting salary distribution plot
SalStat = sns.countplot(data2['SalStat'])
plt.show()

#Histogram of age
sns.distplot(data2['age'], bins=10, kde=False)
plt.show()

#Boxplot for age vs salary
sns.boxplot(x='SalStat', y='age', data=data2)
data2.groupby('SalStat')['age'].median()
plt.show()

#Baased on job type - salary
JobType = sns.countplot(y=data2['JobType'],hue = 'SalStat', data=data2)
plt.show()
job_salstat =pd.crosstab(index = data2["JobType"],columns = data2['SalStat'], margins = True, normalize =  'index')  
print(round(job_salstat*100,1))

#Based on education - salary
Education = sns.countplot(y=data2['EdType'],hue = 'SalStat', data=data2)
plt.show()
EdType_salstat = pd.crosstab(index = data2["EdType"], columns = data2['SalStat'],margins = True,normalize ='index')  
print(round(EdType_salstat*100,1))

#Based on Occupation - salary
Occupation  = sns.countplot(y=data2['occupation'],hue = 'SalStat', data=data2)
plt.show()
occ_salstat = pd.crosstab(index = data2["occupation"], columns =data2['SalStat'],margins = True,normalize = 'index')  
round(occ_salstat*100,1)

#Capital gain
sns.distplot(data2['capitalgain'], bins = 10, kde = False)
plt.show()

#Capital loss
sns.distplot(data2['capitalloss'], bins = 10, kde = False)
plt.show()

#Logistic Regression

# Reindexing the salary status names to 0,1
data2['SalStat']=data2['SalStat'].map({' less than or equal to 50,000':0,' greater than 50,000':1})
print(data2['SalStat'])

new_data=pd.get_dummies(data2, drop_first=True)

# Storing the column names 
columns_list=list(new_data.columns)
print(columns_list)

# Separating the input names from data
features=list(set(columns_list)-set(['SalStat']))
print(features)

# Storing the output values in y
y=new_data['SalStat'].values
print(y)

# Storing the values from input features
x = new_data[features].values
print(x)

# Splitting the data into train and test
train_x,test_x,train_y,test_y = train_test_split(x,y,test_size=0.3, random_state=0)

# Make an instance of the Model
logistic = LogisticRegression()

# Fitting the values for x and y
logistic.fit(train_x,train_y)
logistic.coef_
logistic.intercept_

# Prediction from test data
prediction = logistic.predict(test_x)
print(prediction)

# Confusion matrix
confusion_matrix = confusion_matrix(test_y, prediction)
print(confusion_matrix)

# Calculating the accuracy
accuracy_score=accuracy_score(test_y, prediction)
print(accuracy_score)

# Printing the misclassified values from prediction

print('Misclassified samples: %d' % (test_y != prediction).sum())

#Logistic Regression after removing insignificant variables

# Reindexing the salary status names to 0,1
data3['SalStat']=data3['SalStat'].map({' less than or equal to 50,000':0,' greater than 50,000':1})
print(data3['SalStat'])

cols = ['gender','nativecountry','race','JobType']
new_data = data3.drop(cols,axis = 1)

new_data=pd.get_dummies(new_data, drop_first=True)

# Storing the column names 
columns_list2=list(new_data.columns)
print(columns_list2)

# Separating the input names from data
features2=list(set(columns_list2)-set(['SalStat']))
print(features2)

# Storing the output values in y
y2=new_data['SalStat'].values
print(y2)

# Storing the values from input features
x2 = new_data[features2].values
print(x2)

# Splitting the data into train and test
train_x2,test_x2,train_y2,test_y2 = train_test_split(x2,y2,test_size=0.3, random_state=0)

# Make an instance of the Model
logistic2 = LogisticRegression()

# Fitting the values for x and y
logistic2.fit(train_x2,train_y2)

# Prediction from test data
prediction2 = logistic2.predict(test_x2)

# Printing the misclassified values from prediction
print('Misclassified samples: %d' % (test_y2 != prediction2).sum())
