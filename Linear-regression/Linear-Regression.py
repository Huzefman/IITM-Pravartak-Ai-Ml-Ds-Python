#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 17 18:46:23 2025

@author: huzefa
"""

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import statsmodels.api as sm

sns.set(rc={'figure.figsize':(11.7,8.27)})

cars_data=pd.read_csv('../Datasets/cars_sampled1.csv' )

cars=cars_data.copy()
cars.info()

cars.describe()
pd.set_option('display.float_format', lambda x: '%.3f' % x)
pd.set_option('display.precision',2)
cars.describe()

pd.set_option('display.max_columns', 500)
cars.describe()

np.set_printoptions(suppress=True)

col=['name','dateCrawled','dateCreated','postalCode','lastSeen']
cars=cars.drop(columns=col, axis=1)
cars.drop_duplicates(keep='first',inplace=True)

cars = cars[
        (cars.yearOfRegistration <= 2018) 
      & (cars.yearOfRegistration >= 1950) 
      & (cars.price >= 100) 
      & (cars.price <= 150000) 
      & (cars.powerPS >= 10) 
      & (cars.powerPS <= 500)]

cars['monthOfRegistration']/=12
cars['Age']=(2018-cars['yearOfRegistration'])+cars['monthOfRegistration']
cars['Age']=round(cars['Age'],2)
cars['Age'].describe()
cars=cars.drop(columns=['yearOfRegistration','monthOfRegistration'], axis=1)

col=['seller','offerType','abtest']
cars=cars.drop(columns=col, axis=1)
cars_copy=cars.copy()

cars_select1=cars.select_dtypes(exclude=[object])
correlation=cars_select1.corr()
round(correlation,3)   
cars_select1.corr().loc[:,'price'].abs().sort_values(ascending=False)[1:]                    

cars_omit=cars.dropna(axis=0)
cars_omit.to_csv('omitted_data_cars_sampled.csv')

cars_omit_data=pd.read_csv('omitted_data_cars_sampled.csv')
cars_omit_data=cars_omit_data.drop('Unnamed: 0',axis=1)

x1 = cars_omit_data.filter(['powerPS','kilometer','Age'],axis=1)
y1= cars_omit_data.filter(['price'],axis=1)

X_train, X_test, y_train, y_test = train_test_split(x1, y1, test_size=0.3, random_state = 3)
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

def calculateVIF(data):
    features = list(data.columns)
    num_features = len(features)

    model = LinearRegression()

    result = pd.DataFrame(index = ['VIF'], columns = features)
    result = result.fillna(0)

    for ite in range(num_features):
        x_features = features[:]
        y_featue = features[ite]
        x_features.remove(y_featue)     
        model.fit(data[x_features], data[y_featue])
        result[y_featue] = 1/(1 - model.score(data[x_features], data[y_featue]))

    return result

def rmse(test_y,predicted_y):
    rmse_test=np.sqrt(mean_squared_error(test_y, predicted_y))
    #for base rmse
    base_pred = np.repeat(np.mean(test_y), len(test_y))
    rmse_base = np.sqrt(mean_squared_error((test_y), base_pred))
    values={'RMSE-test from model':rmse_test,'Base RMSE':rmse_base}
    return values

X_train2 = sm.add_constant(X_train)
model_lin1 = sm.OLS(y_train, X_train2)
results1=model_lin1.fit()
print(results1.summary())

X_test=sm.add_constant(X_test)
cars_predictions_lin1_test = results1.predict(X_test)

vif_val=calculateVIF(X_train)
vif_val=vif_val.transpose()

rmse(y_test,cars_predictions_lin1_test)

cars_predictions_lin1_train = results1.predict(X_train2)

residuals=y_train.iloc[:,0]-cars_predictions_lin1_train

sns.regplot(x=cars_predictions_lin1_train,y=residuals)
plt.xlabel("Fitted values")
plt.ylabel("Residuals")
plt.title('Residual plot')
plt.show()

sm.qqplot(residuals)
plt.title("Normal Q-Q Plot")
plt.show()

# Plotting the variable price
prices = pd.DataFrame({"1. Before":y1.iloc[:,0], "2. After":np.log(y1.iloc[:,0])})
prices.hist()
plt.show()

# Plotting the variable price
prices = pd.DataFrame({"1. Before":y_train.iloc[:,0], "2. After":np.log(y_train.iloc[:,0])})
prices.hist()
plt.show()

# Plotting the variable price
prices = pd.DataFrame({"1. Before":y_test.iloc[:,0], "2. After":np.log(y_test.iloc[:,0])})
prices.hist()
plt.show()

y2=np.log(y1)
y_train_log,y_test_log=train_test_split(y2, test_size=0.3, random_state = 3)

X_train2 = sm.add_constant(X_train)
model_lin2 = sm.OLS(y_train_log, X_train2)
results2=model_lin2.fit()
print(results2.summary())

X_test=sm.add_constant(X_test)
cars_predictions_lin2_test = results2.predict(X_test)

def rmse_log(test_y,predicted_y):
    t1=np.exp(test_y)
    t2=np.exp(predicted_y)
    rmse_test=np.sqrt(mean_squared_error(t1,t2))
    
    #for base rmse
    base_pred = np.repeat(np.mean(t1), len(t1))
    rmse_base = np.sqrt(mean_squared_error(t1, base_pred))
    values={'RMSE-test from model':rmse_test,'Base RMSE':rmse_base}
    return values

# Model evaluation on predicted and test 
rmse_log(y_test_log,cars_predictions_lin2_test)

cars_predictions_lin2_train = results2.predict(X_train2)

residuals=y_train_log.iloc[:,0]-cars_predictions_lin2_train

# Residual plot
sns.regplot(x=cars_predictions_lin2_train,y=residuals)
plt.xlabel("Fitted values")
plt.ylabel("Residuals")
plt.title('Residual plot')
plt.show()

# QQ plot
sm.qqplot(residuals)
plt.title("Normal Q-Q Plot")
plt.show()

x1 = cars_omit_data.drop(['price','model','brand'], axis='columns', inplace=False)
x1=pd.get_dummies(x1,drop_first=True) 
X_train, X_test = train_test_split(x1,test_size=0.3, random_state = 3)

print(type(y_train_log))
# If it's a DataFrame, convert to a Series:
# Ensure y is a Series and numeric
if isinstance(y_train_log, pd.DataFrame):
    y_train_log = y_train_log.iloc[:, 0]
y_train_log = pd.to_numeric(y_train_log, errors='coerce')

# Remove rows with any NaNs
mask = X_train2.notnull().all(axis=1) & y_train_log.notnull()
X_train2_clean = X_train2[mask]
y_train_log_clean = y_train_log[mask]

# Fit the model
model_lin3 = sm.OLS(y_train_log_clean, X_train2_clean)
results3 = model_lin3.fit()
print(results3.summary())

# First, ensure you add the constant to the test data
X_test_simple = sm.add_constant(X_test)

# Now select only the columns used in the model
X_test_simple = X_test_simple[['const', 'powerPS', 'kilometer', 'Age']]

cars_predictions_lin3_test = results3.predict(X_test_simple)

# Model evaluation on predicted and test 
rmse_log(y_test_log,cars_predictions_lin3_test)

cars_predictions_lin3_train = results3.predict(X_train2)

residuals = y_train_log - cars_predictions_lin3_train

# Residual plot
sns.regplot(x=cars_predictions_lin3_train,y=residuals)
plt.xlabel("Fitted values")
plt.ylabel("Residuals")
plt.title('Residual plot')
plt.show()

# QQ plot
sm.qqplot(residuals)
plt.title("Normal Q-Q Plot")
plt.show()