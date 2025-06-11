#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 11 09:48:16 2025

@author: huzefa
"""

import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

#Changing Working Directory
os.chdir('/Users/huzefa/IITM-Pravartak-Ai-Ml-Ds-Python/Week4/Data_Preparation')

#Importing all the necessasry datasets
acDetails=pd.read_table('../../DataSets/acDetails.txt',index_col=0,sep='\t')
demoDetails=pd.read_csv('../../DataSets/demoDetails.csv',index_col=0)
serviceDetails=pd.read_csv('../../DataSets/serviceDetails.csv',index_col=0)

#Checking if determined common attribute is actually common and consistent across all the dataframes
acDetails.customerID.equals(demoDetails.customerID)
demoDetails.customerID.equals(serviceDetails.customerID)

#Checking for any duplicate record in any file (Common Unique attribute 'customer_ID') can be used to check
#Checking count of duplicate records if any present
print(len(np.unique(acDetails['customerID'])))
print(len(np.unique(demoDetails['customerID'])))
print(len(np.unique(serviceDetails['customerID'])))

#Determining the duplicate record
acDetails[acDetails.duplicated(['customerID'],keep=False)]
demoDetails[demoDetails.duplicated(['customerID'],keep=False)]
serviceDetails[serviceDetails.duplicated(['customerID'],keep=False)]

#Removing duplicate record
acDetails=acDetails.drop_duplicates()
demoDetails=demoDetails.drop_duplicates()
serviceDetails=serviceDetails.drop_duplicates()

#Now joining the dataframes into one dataframe
churn=pd.merge(demoDetails, acDetails, on='customerID')
churn=pd.merge(churn, serviceDetails, on='customerID')
churn1=churn.copy()

#Getting basic information of churn1 about the count of null values and datatypes of different variables
churn1.info()
churn1.isnull().sum()
#From the obtained info I found 'SeniorCitizen(null values, incorrect dtype)', 'tenure(incorrect dtype', 'MonthlyCharges(null values)', 'TotalCharges(null values)' as dirty or unclean and have to do further operations on them

# #Checking all the variables unique value to reverify if any other dirty data is present in any column
# np.unique(churn1['customerID']) #Some values contain irregular pattern
# np.unique(churn1['gender']) #All good
# np.unique(churn1['SeniorCitizen'],return_counts=True) #Contains nan values
# np.unique(churn1['Partner']) #All good
# np.unique(churn1['Dependents']) #Contains abnormal values '1@#'
# np.unique(churn1['tenure']) #Incorrect datatype as some values have string instead of int
# np.unique(churn1['Contract']) #All good
# np.unique(churn1['PaperlessBilling']) #All good
# np.unique(churn1['PaymentMethod']) #All good
# np.unique(churn1['MonthlyCharges']) #Contains nan values
# np.unique(churn1['TotalCharges']) #Contains nan values
# np.unique(churn1['PhoneService']) #All good
# np.unique(churn1['MultipleLines']) #All good
# np.unique(churn1['InternetService']) #All good
# np.unique(churn1['OnlineSecurity']) #All good
# np.unique(churn1['DeviceProtection']) #All good
# np.unique(churn1['TechSupport']) #All good
# np.unique(churn1['StreamingTV']) #All good
# np.unique(churn1['StreamingMovies']) #All good
# np.unique(churn1['Churn']) #All good

#Using a quick approach to quickly get the above information about objects using lambda function
categorical_data=churn1.select_dtypes(include=['object']).copy()
categorical_data=categorical_data.drop(['customerID','tenure'],axis=1)#dropping these because they overpopulate the information
frequencies=categorical_data.apply(lambda x: x.value_counts()).T.stack()
print(frequencies)



#Cleaning and making 'customerID' consistent
#Length of 'customerID' must be 10 and pattern as '[nnnn]-[ccccc]' first 4 numbers + - + last 5 characters
len_ind=[i for i,value in enumerate(churn1.customerID) if len(value)!=10] #All have length = 10

import re

pattern='^[0-9]{4,4}-[A-Z]{5,5}'
type(pattern)
p=re.compile(pattern)
type(p)

q=[i for i,value in enumerate(churn1.customerID) if p.match(str(value))==None]
print(q) #At indices 2,3,4,6 pattern doesn't match
#Two types of anomalies present 1.) '/' insted of '-' 2.) first 5 characters + - + last 4 digits
fp1=re.compile('^[A-Z]{5,5}-[0-9]{4,4}')
fp2=re.compile('^[0-9]{4,4}/[A-Z]{5,5}')

for i in q:
    false_str=str(churn1.customerID[i])
    if(fp1.match(false_str)):
        str_splits=false_str.split('-')
        churn1.customerID[i]=str_splits[1]+'-'+str_splits[0]
    elif(fp2.match(false_str)):
        str_splits=false_str.split('/')
        churn1.customerID[i]=false_str.replace('/','-')
        
#Rechecking if customerID is now cleaned or consistent or not
np.unique(churn1['customerID'])

#Cleaning 'SeniorCitizen' resolving nan values (using mode)
churn1['SeniorCitizen'].fillna(churn1['SeniorCitizen'].mode()[0],inplace=True)
churn1.isnull().sum() #All values filled
churn1['SeniorCitizen']=churn1.SeniorCitizen.astype(int)

#Cleaning 'Dependents' removing abnormal values
pd.crosstab(index=churn1['Dependents'], columns='count')
churn1['Dependents']=churn1['Dependents'].replace('1@#','No')

#Cleaning 'tenure' and making it consistent int datatype
churn1['tenure']=churn1.tenure.replace("Four",4)
churn1['tenure']=churn1.tenure.replace("One",1)
churn1['tenure']=churn1.tenure.astype(int)
np.unique(churn1['tenure'])

#Cleaning 'MonthlyCharges' filling nan values
churn1.describe()
sns.boxplot(x=churn1['MonthlyCharges'],y=churn1['Churn'])
plt.show() #The plot show that the mean of customers who have churned and those who haven't anre not same, hence need to calculate seperate means for both categories
churn1.groupby('Churn')['MonthlyCharges'].mean()
churn1['MonthlyCharges']=churn1.groupby('Churn')['MonthlyCharges'].transform(lambda x:x.fillna(x.mean()))

#Similarly cleaning 'TotalCharges' and doing same as above 'MonthlyCharges'
sns.boxenplot(x=churn1['TotalCharges'],y=churn1['Churn'])
plt.show()
churn1.groupby('Churn')['TotalCharges'].mean()
churn1['TotalCharges']=churn1.groupby('Churn')['TotalCharges'].transform(lambda x:x.fillna(x.mean()))
churn1.info()

#I also need to deal with outliers if present in any data
sns.boxenplot(x=churn1['tenure'])
plt.show() #Outliers present so need to replace them with the median for any value greater than 500 as outlier lie after that in this case
churn1['tenure']=np.where(churn1['tenure']>=500,churn1['tenure'].median(),churn1['tenure'])
sns.boxenplot(y=churn1['tenure'])
plt.show()

#Also there exists a logical fallacy about the 'InternetService' and its allied services. It may be possible if 'InternetServices' may be marked as 'No' but its allied services may be marked 'Yes', can be vice versa also
#To solve this 2 approaches are there 1.) Brute force where if 'InternetService' marked 'No' then all the allied services also mark them as 'No'. 2) Logical Approach where if 'InternetService' marked 'No' but 2 or more allied services marked as 'Yes' this may indicate that possible error occured in 'InternetService' and it is marked as 'Yes'
#Solving using logical approach
y=churn1[(churn1.InternetService=='No')] #Subsetting the 'No' 'InternService'
z=y.iloc[:,13:20] #Further subsetting the 'No' 'InternService' and its allied services

for i, row in z.iterrows():
    yes_cnt=row.str.count('Yes').sum()
    if(yes_cnt>=2):
        z.loc[i].InternetService='Yes'
    else:
        z.loc[i,:]='No interntet service'
        
#Demo for Random sampling
import random
p1=list(range(1,20))
print(p1)

#Without replacement
srswor=random.sample(population=p1,k=10)
print(srswor)
#With replacement
srswr=random.choices(population=p1,k=10)
print(srswr)
churn1.to_csv('Clean_Churn_Data.csv',index=False)


