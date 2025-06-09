#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun  7 18:39:33 2025

@author: huzefa
"""

import pandas as pd

print("This is demo for reading data from csv")

data_csv1=pd.read_csv('../../DataSets/Iris_data_sample.csv')
data_csv2=pd.read_csv('../../DataSets/Iris_data_sample.csv',index_col=0)
data_csv3=pd.read_csv('../../DataSets/Iris_data_sample.csv',index_col=0,na_values=["??","###"])

print("This is demo for reading data from excel")

data_xslv1=pd.read_excel('../../DataSets/Iris_data_sample.xlsx')
data_xslv2=pd.read_excel('../../DataSets/Iris_data_sample.xlsx',index_col=0)
data_xslv3=pd.read_excel('../../DataSets/Iris_data_sample.xlsx',index_col=0,na_values=["??","###"])

print("This is demo for reading data from txt")

data_txt1=pd.read_table('../../DataSets/Iris_data_sample.txt')
data_txt2=pd.read_table('../../DataSets/Iris_data_sample.txt',index_col=0, sep=' ')
data_txt3=pd.read_csv('../../DataSets/Iris_data_sample.txt',index_col=0, sep=' ')

print("This is demo for shallow copy")

data_csv4=data_csv1.copy(deep=False)

print("This is demo for shallow copy")

data_csv5=data_csv1.copy(deep=False)
data_csv6=data_csv1


print("Attributes of Data: ")

print(data_csv3.index)
print(data_csv3.size)
print(data_csv3.columns)
print(data_csv3.shape)
print(data_csv3.memory_usage())
print(data_csv3.ndim)

print("Indexing and selecting data")

print(data_csv3.head(5))
print(data_csv3.tail(5))
print(data_csv3.at[7,'SepalLengthCm'])
print(data_csv3.iat[5,3])
print(data_csv3.loc[:,'SepalLengthCm'])

print("Datatypes")

print(data_csv3.dtypes)
print(data_csv3.info)

print("To convert into other datatypes")

data_csv6['SepalLengthCm']=data_csv6['SepalLengthCm'].astype('object')

print("To replace a value")

data_csv6['SepalLengthCm']=data_csv6['SepalLengthCm'].replace('three',3)

print("To get total bytes consumed")

print(data_csv3['SepalLengthCm'].nbytes)

print("To check missing values")

print(data_csv1.isnull().sum())

print("To count series of unique values")

print(data_csv3['SepalLengthCm'].value_counts())