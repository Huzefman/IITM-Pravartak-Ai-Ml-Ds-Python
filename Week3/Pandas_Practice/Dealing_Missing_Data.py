#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun  8 10:59:38 2025

@author: huzefa
"""

import pandas as pd

cars_data=pd.read_csv('../../DataSets/Toyota.csv',index_col=0,na_values=['??','????'])
cars_data1=cars_data.copy()

print("Getting no of missing values in each column")
print(cars_data1.isna().sum())

print("Getting the rows which have atleast one column value missing")

missing=cars_data1[cars_data1.isnull().any(axis=1)]

print("Getting information stats about give Data")
description=cars_data1.describe()

# print("Filling numerical missing values with mean or median")
# cars_data1['Age']=cars_data1['Age'].fillna(cars_data1['Age'].mean())
# cars_data1['KM']=cars_data1['KM'].fillna(cars_data1['KM'].median())
# cars_data1['HP']=cars_data1['HP'].fillna(cars_data['HP'].mean())

# print("Filling Categorical missing values with mode")
# cars_data1['FuelType']=cars_data1['FuelType'].fillna(cars_data1['FuelType'].mode()[0])
# cars_data1['MetColor']=cars_data1['MetColor'].fillna(cars_data1['MetColor'].mode()[0])

print("Filling all the missing values in one shot using lambda function")
print("Filling all missing values in one shot using lambda function")
cars_data1=cars_data1.apply(lambda x: x.fillna(x.mean()) if x.dtype=='float' else x.fillna(x.mode()[0]))

print(cars_data1.isna().sum())