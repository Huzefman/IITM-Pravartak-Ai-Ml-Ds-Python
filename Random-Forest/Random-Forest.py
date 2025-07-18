#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 18 18:20:02 2025

@author: huzefa
"""

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import RandomizedSearchCV
import statsmodels.api as sm

cars_omit_data=pd.read_csv('../Linear-Regression/omitted_data_cars_sampled.csv')
cars_omit_data=cars_omit_data.drop('Unnamed: 0',axis=1)

x1 = cars_omit_data.drop(['price','model','brand'], axis='columns', inplace=False)
x1=pd.get_dummies(x1,drop_first=True) 
y1= cars_omit_data.filter(['price'],axis=1)
# Transforming prices to log
y2=np.log(y1)

X_train, X_test, y_train_log,y_test_log = train_test_split(x1, y2, test_size=0.3, random_state = 3)
print(X_train.shape, X_test.shape, y_train_log.shape, y_test_log.shape)

def rmse_log(test_y,predicted_y):
    t1=np.exp(test_y)
    t2=np.exp(predicted_y)
    rmse_test=np.sqrt(mean_squared_error(t1,t2))
    #for base rmse
    base_pred = np.repeat(np.mean(t1), len(t1))
    rmse_base = np.sqrt(mean_squared_error(t1, base_pred))
    values={'RMSE-test from model':rmse_test,'Base RMSE':rmse_base}
    return values

rf = RandomForestRegressor(n_estimators = 220,max_depth=87)

# Model
model_rf1=rf.fit(X_train,y_train_log)

# Predicting model on test and train set
cars_predictions_rf1_test = rf.predict(X_test)

# RMSE
rmse_log(y_test_log,cars_predictions_rf1_test)

# Rsquared
model_rf1.score(X_train,y_train_log)

## Number of trees in random forest
n_estimators = [int(x) for x in np.linspace(start = 10, stop = 600,num = 15)]
print(n_estimators)

## Maximum number of levels in tree
max_depth = [int(x) for x in np.linspace(10, 110, num = 10)]


## Minimum number of samples required to split a node
min_samples_split = np.arange(100,1100,100)



random_grid = {'n_estimators': n_estimators,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split}

print(random_grid)

rf_for_tuning = RandomForestRegressor()

rf_random = RandomizedSearchCV(estimator = rf_for_tuning, 
                            param_distributions = random_grid, 
                            n_iter = 100,cv = 3, verbose=2, random_state=1)

## Fit the random search model
rf_random.fit(X_train,y_train_log)
print(rf_random.best_params_)

## finding the best model
rf_model_best = rf_random.best_estimator_
print(rf_model_best)

# predicting with the test data on best model
predictions_best = rf_model_best.predict(X_test)
predictions_best_train=rf_model_best.predict(X_train)