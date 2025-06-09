#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun  8 10:13:16 2025

@author: huzefa
"""

import pandas as pd

cars_data=pd.read_csv('Toyota.csv',index_col=0,na_values=['????','??'])
cars_data1=cars_data.copy()

print("Creating a simple frequency table corsstabulation")
fuelType_count=pd.crosstab(index=cars_data1['FuelType'], columns='count',dropna=True)

print("Creating a two way table - joint probability")
automatic_fuelType=pd.crosstab(index=cars_data1['Automatic'], columns=cars_data1['FuelType'],dropna=True,normalize=True)

print("Creating a two way table - marginal probabitliy")
automatic_fuelType1=pd.crosstab(index=cars_data1['Automatic'], columns=cars_data1['FuelType'],dropna=True,margins=True,normalize=True)

print("Creating a two way table - conditional proabbility")
automatic_fuelType2=pd.crosstab(index=cars_data1['Automatic'], columns=cars_data1['FuelType'],dropna=True,margins=True,normalize='index')
automatic_fuelType3=pd.crosstab(index=cars_data1['Automatic'], columns=cars_data1['FuelType'],dropna=True,margins=True,normalize='columns')

print("Obtaining Correlation Matrix")
#first getting all the numerical data from original data as correlation can be only performed on numerical data not on objects or categorical data
numericalData=cars_data1.select_dtypes(exclude=[object])
#second check the dimension of obtained numericalData
print(numericalData.shape)
#now the correlation matrix
corr_matrix=numericalData.corr()