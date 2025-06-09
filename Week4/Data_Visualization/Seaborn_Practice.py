#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun  9 14:01:00 2025

@author: huzefa
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

#importing data
cars_data=pd.read_csv('../../DataSets/Toyota.csv',index_col=0)
cars_data1=cars_data.copy()
cars_data1.dropna(axis=0,inplace=True)
cars_data1.info()


#Scatter Plot
sns.set(style="darkgrid")
sns.regplot(x=cars_data1['Age'],y=cars_data1['Price'], fit_reg=False, marker='*')
plt.show()

#Scatter Plot with Group
sns.lmplot(x='Age', y='Price', data=cars_data1, fit_reg=False, hue='HP', legend=True, palette="Set1")
plt.show()

#Histogram
sns.displot(cars_data1['Age'], kde=False, bins=5)
plt.show()

#Bar Plot
sns.countplot(x='FuelType', data=cars_data1)
plt.show()

#Bar Plot with Group
sns.countplot(x='FuelType', data=cars_data1, hue='Automatic')
plt.show()

#Box and Whiskers Plot
sns.boxenplot(y=cars_data1['Price'])
plt.show()

#Categorical vs Numerical Box and Whiskers Plot
sns.boxenplot(x=cars_data1['FuelType'],y=cars_data1['Price'])
plt.show()

#Grouped Box Plot
sns.boxplot(x=cars_data1['FuelType'],y=cars_data1['Price'], hue=cars_data1["Automatic"])
plt.show()

#Box Whiskers Plot and Histogram in same Plot by Splitting Window
f,(ax_box,ax_hist)=plt.subplots(2,gridspec_kw={"height_ratios":(.50,.50)})
sns.boxenplot(cars_data1['Price'], ax=ax_box)
sns.histplot(cars_data1["Price"],ax=ax_hist, kde=False)
plt.show()

#Pariwise Plots
sns.pairplot(cars_data1, kind="scatter",hue="FuelType")
plt.show()
