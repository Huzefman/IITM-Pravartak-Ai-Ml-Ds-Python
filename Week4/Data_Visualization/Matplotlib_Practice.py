#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun  9 10:55:15 2025

@author: huzefa
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#importing the data
cars_data=pd.read_csv('Toyota.csv')
cars_data1=cars_data.copy()

#removing all the nan values
cars_data1.dropna(axis=0,inplace=True)

#Scatter plot
plt.scatter(cars_data1['Age'], cars_data1['KM'], c='red')
plt.title("Age vs KM")
plt.xlabel('Age')
plt.ylabel('KM')
plt.show()

#Histogram
plt.hist(cars_data1['KM'], color='green', edgecolor='white', bins=5)
plt.title('No of cars in Range of KM')
plt.xlabel('KM')
plt.ylabel('Count')
plt.show()

#Bar Plot
fuel_counts=cars_data1['FuelType'].value_counts()
index=fuel_counts.index.tolist()
#fueltype=fuel_counts.index.tolist()
#index=np.arange(len(fueltype))
count=fuel_counts.values.tolist()
plt.bar(index, count, color=['red','blue','cyan'])
plt.title('No of cars using different fuels')
plt.xlabel('Fuel Type')
plt.ylabel('Count')
#plt.xticks(index, fueltype, rotation=90)
plt.xticks(index, rotation=90)

plt.show()
