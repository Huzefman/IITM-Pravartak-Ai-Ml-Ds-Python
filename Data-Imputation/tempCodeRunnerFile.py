#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 27 17:02:30 2025

@author: huzefa
"""

import pandas as pd
import numpy as np
import os

# Change working directory to a new path
os.chdir('/Users/huzefa/IITM-Pravartak-Ai-Ml-Ds-Python/Data-Imputation')

# Verify the change
print("Current working directory:", os.getcwd())

#Importing Data
data=pd.read_csv('../DataSets/GTPvar.csv',index_col=0)

#Checking missing values columnwise
data.isnull().sum(axis=1)

#Creating new variable NApresent to contain all the rows' missing values sum
data['NApresent']=data.isnull().sum(axis=1)

#Extracting rows with no null values
df=data[data.NApresent==0]

#Dropping NApresent column from df
df=df.drop('NApresent',axis=1)

#Converting np to numpy array
df_mat=df.to_numpy()

#Rank of obtained matrix
np.linalg.matrix_rank(df_mat)

#SVD decomposition
v,s,u=np.linalg.svd(df_mat.T)

#Setting tolerance
tol=1e-8

#Removing columns that are lesser than the tolerance
rank=min(df_mat.shape)-np.abs(s)[::-1].searchsorted(tol)

#Choosing the null space relation
A=v[:,rank:]
A=A.T
print(A)