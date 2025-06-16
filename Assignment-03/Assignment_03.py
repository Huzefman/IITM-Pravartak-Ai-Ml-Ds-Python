#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun  8 12:05:46 2025

@author: huzefa
"""

import pandas as pd

tips_data=pd.read_csv('../Datasets/Tips.csv',index_col=0)
tips_data1=tips_data.copy()

#To get Numerical data from give dataset
numerical_data=tips_data1.select_dtypes(exclude=[object])

#To get Correlation Matrix
corr_matrix=numerical_data.corr()

#To get basic stats of tips_data
description=tips_data1.describe()

#To get count missing values in each column
print(tips_data1.isna().sum())

#Getting a subset of rows with atleast one missing values
missing=tips_data1[tips_data1.isna().any(axis=1)]

#Filling numerical values using lambda function
tips_data1=tips_data1.apply(lambda x: x.fillna(x.mean()) if x.dtype=='float' else x.fillna(x.value_counts().index[0]))

print(tips_data1.isna().sum())
