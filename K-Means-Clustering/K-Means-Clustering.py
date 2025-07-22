#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 21 19:53:19 2025

@author: huzefa
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import cluster
from sklearn.preprocessing import StandardScaler
import copy as cp

#Reading data
data=pd.read_excel('../Datasets/tripDetails.xlsx')
print(data.head())

#Dropping TripID column
data=data.drop(['TripID'],axis=1)
print(data.head())

#Taking column name as features
features=list(data.columns)

#Creating a new dataframe for units of the data columns
units=['kms','kmph','kmps','mins','counts','mins','counts']

#Mapping features to units
feature_units=dict(zip(features,units))
print(feature_units)

#Data types of variables
data.info()
data.describe()

#Histogram of each features
for i in features:
    data[i].plot(kind='hist', bins=15)
    plt.title(i)
    plt.xlabel(feature_units[i])
    plt.show()

#Checking relationships between different features
correlation=data.corr()
print(correlation)

#Visualizing correlation using heatmaps
sns.heatmap(np.abs(correlation), xticklabels=correlation.columns,yticklabels=correlation.columns)
plt.show()

#Visualizing scatter data
sns.pairplot(data)
plt.show()

#Now scaling all the values to same level
data2=data.copy()
data2=StandardScaler().fit_transform(data2.values)
data2=pd.DataFrame(data2,columns=features)

#Applying k-means clustering for different sequential values and plotting an elbow plot to get the most optimum k value
distortions=[]
for i in range(1,11):
    km=cluster.KMeans(n_clusters=i,init='k-means++',n_init=10,max_iter=300,random_state=100)
    km.fit(data2.values)
    distortions.append(km.inertia_)

plt.figure(figsize = (7,7))
plt.plot(range(1,11), distortions, marker='o')
plt.title('ELBOW PLOT')
plt.xlabel('Number of clusters')
plt.ylabel('Distortion')
plt.show()

#Taking k=3 is best value obtained
k = 3
km3 = cluster.KMeans(n_clusters=k,
init='k-means++',
n_init = 10,
max_iter = 300,
random_state = 100)
km3.fit(data2.values)

#Obtaining lables for each row to map to their clusters respectively
labels = km3.labels_
Ccenters = km3.cluster_centers_
data2['labels'] = labels
data2['labels'] = data2['labels'].astype('str')
print(data2['labels'])

#Pairplot after clustering
sns.pairplot(data2, x_vars = features, y_vars = features, hue='labels', diag_kind='kde')
plt.show()

#Clusterwise feature analysis using original data
c_df = pd.concat([data[data2['labels']=='0'].mean(),
data[data2['labels']=='1'].mean(),
data[data2['labels']=='2'].mean()],
axis=1)
c_df.columns = ['cluster1','cluster2','cluster3']
print(c_df)

#Using obtained instights about clusters, it can be infered that the clusters may belong to following practical categories
'''
Cluster1 is distinguised by comparatively very high values for Brakes,
IdlingTime, Honking, low MaxSpeed and TripLength
This is indicative of intercity travel during peak hours

MaxSpeed, MostFreqSpeed and TripDuration is higher for cluster2 than cluster
1 and 3
Cluster2 is is indicative of highway trips

Cluster3 is indicative of city trips during non-peak hours
''' 
#Assigning the names to clusters
triptype = ['Intercity-Peak hours','Highway','Intercity-Non-peak hours']
data['labels'] = labels
data['labels'] = data['labels'].map({0:triptype[0],1:triptype[1],2:triptype[2]})
print(data.head())