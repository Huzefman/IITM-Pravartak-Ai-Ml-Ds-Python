#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 18 10:12:28 2025

@author: huzefa
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

#Importing the dataset
cars_data=pd.read_csv('../Datasets/cars_sampled.csv')

#Getting brief information about the data variables
cars_data.info()

#Converting variables' inappropriate dtype to appropriate dtypes
cars_data['dateCrawled']=pd.to_datetime(cars_data['dateCrawled'],format='%d/%m/%Y %H:%M', errors='coerce')
cars_data['dateCreated']=pd.to_datetime(cars_data['dateCreated'],format='%d/%m/%Y %H:%M', errors='coerce')
cars_data['lastSeen']=pd.to_datetime(cars_data['lastSeen'],format='%d/%m/%Y %H:%M', errors='coerce')
cars_data.info()


#Checking for any duplicate data
cars_data.duplicated().sum()
duplicate=cars_data[cars_data.duplicated(keep=False)]
#Dropping duplicates
cars_data=cars_data.drop_duplicates()
cars_data.info()

#Checking for all the object variables' catergories and respective counts
categorical_data=cars_data.select_dtypes(include=['object']).copy()
categorical_data=categorical_data.drop(['name'],axis=1)
frequencies=categorical_data.apply(lambda x: x.value_counts(dropna=False)).T.stack()
print(frequencies)


#Checking for number of nan values in vehicleType
cars_data.vehicleType.isnull().sum()
#Filling the missing values with modal value
cars_data['vehicleType']=cars_data['vehicleType'].fillna(cars_data['vehicleType'].mode()[0])
cars_data.vehicleType.isnull().sum()
cars_data.info()

#Checking basic description of numerical data and checking for logic
dec=cars_data.describe()

#Filling nan values of fuelType considering brand
pd.crosstab(index=cars_data['brand'], columns=cars_data['fuelType'],dropna=False)
cars_data['fuelType']=cars_data.groupby('brand')['fuelType'].transform(lambda x: x.fillna(x.mode()[0]))
pd.crosstab(index=cars_data['brand'], columns=cars_data['fuelType'],dropna=False)

#Filling nan values of gearbox by considering fuelType
pd.crosstab(index=cars_data['gearbox'], columns=cars_data['fuelType'],dropna=False)
#Grouping gearbox as fuelType and filling nan values with respective modal values
cars_data['gearbox']=cars_data.groupby('fuelType')['gearbox'].transform(lambda x: x.fillna(x.mode()[0]))
cars_data.info()

#Filling nan values of model considering brand
pd.crosstab(index=cars_data['brand'], columns=cars_data['model'],dropna=False)
def fill_with_mode(x):
    mode = x.mode()
    if not mode.empty:
        return x.fillna(mode[0])
    else:
        return x

cars_data['model'] = cars_data.groupby(['brand', 'vehicleType'])['model'].transform(fill_with_mode)
cars_data['model']=cars_data['model'].fillna(cars_data['model'].mode()[0])
cars_data.info()

#Filling nan values of norepaireddamaged with mode
cars_data['notRepairedDamage']=cars_data['notRepairedDamage'].fillna(cars_data['notRepairedDamage'].mode()[0])
cars_data.info()

#Converting to csv file
cars_data.to_csv('clean_cars_sampled.csv',index=False)