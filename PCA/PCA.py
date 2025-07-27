#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul 27 12:29:52 2025

@author: huzefa
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

#Importing data
irisData=pd.read_csv('../Datasets/Iris_data.csv')

#Info of data
irisData.info()
print(irisData.head())

#Dropping Id column
irisData=irisData.drop('Id',axis=1)
print(irisData.head())

#Sorting the columns alphabetically
inputColumns=list(irisData.iloc[:,:4].columns)
inputColumns.sort()
inputData=irisData[inputColumns]
print(inputData.head())

#Plotting a scatter plot of 2 variables from irisData
plt.subplots(figsize=(8,6))
sns.scatterplot(x='SepalLengthCm',y='SepalWidthCm',hue='Species',data=irisData)
plt.show()

#Scaling data to standard
scaler=StandardScaler(with_std=False)
inputData=scaler.fit_transform(inputData)
inputData=pd.DataFrame(inputData,columns=inputColumns)
print(inputData.head())

#Computing covariance using scaled data
covMat=inputData.cov()
print(covMat)

#Computing eigenvectors and eigenvalues of covariance matrix
eigVal, eigVec= np.linalg.eig(covMat.values)
eigPairs = [(np.abs(eigVal[i]), eigVec[:,i])for i in range(len(eigVal))]
print(eigPairs)

#Sorting eigPairs in descending order based on the eigen values
eigPairs.sort(key = lambda x: x[0], reverse=True)
#false for ascending order
print('Eigenvalues in descending order:')
for i in eigPairs:
    print(i[0])
    
#Setting threshold as '95% variance'
threshold = 0.95
#Computing number of PCS required to captured specified variance
print('Explained variance in percentage:\n')
cumulativeVariance = 0.0
count = 0
eigvSum = np.sum(eigVal)
for i,j in enumerate(eigPairs):
    varianceExplained = (j[0]/eigvSum).real
    print('eigenvalue {}: {}'.format(i+1, varianceExplained*100 ))
    cumulativeVariance += varianceExplained
    count = count+1
    if (cumulativeVariance>=threshold):
        break
print('\nCumulative variance=',cumulativeVariance*100)
print('Total no. of eig vecs =',len(eigVec),'\nselected no. of eig vecs =',count)

#Selecting required PCs based on the count - projection matrix w=d*k
reducedDimension = np.zeros((len(eigVec),count))
for i in range(count):
    reducedDimension[:,i]= eigPairs[i][1]
    
#Projecting the scaled data onto the reduced space (using eigen vectors)
projectedData = inputData.values.dot(reducedDimension)
projectedDataframe = pd.DataFrame(projectedData,
columns=['PC1','PC2'])
projectedDataframeWithClassInfo = pd.concat([projectedDataframe,
irisData.Species],axis=1)

#Plotting PCs
plt.subplots(figsize=(8,6))
sns.scatterplot(x='PC1', y='PC2', hue='Species', data=projectedDataframeWithClassInfo)
plt.show()

#Using SVD
u, s, v = np.linalg.svd(inputData) #decomposing using SVD
expVar=s**2/np.sum(s**2)*100 # Explained variance by each eigen value/PC
pc=irisData[inputColumns].dot(v.T) # Rotating and trnsforming from sample space to feature space
pc.columns=['PC1','PC2','PC3','PC4']
pc['Species']= irisData.Species
print(pc.head())

#Plotting PCs usnig SVD
plt.subplots(figsize=(8,6))
sns.scatterplot(x='PC1', y='PC2', hue='Species', data=pc)
plt.show()

#Directly using PCA function from sklearn
# Choosing the extent of variance to be covered by the PCs
PCASklearn = PCA(n_components=0.95)
# Transforming the iris data input_columns
ProjectedDataSklearn= PCASklearn.fit_transform(irisData.iloc[:,:4])
# Storing the PCs in the data frame
ProjectedDataSklearnDf = pd.DataFrame(ProjectedDataSklearn,columns=['PC1','PC2'])

# Storing the PCs in the data frame along with class label
ProjectedDataSklearnDfWithClassInfo=pd.concat([ProjectedDataSklearnDf,
irisData.Species],axis=1)
print('Explained variance :\n')
print(PCASklearn.explained_variance_ratio_)

#Plotting PCs using direct PCA function values
plt.subplots(figsize=(8,6))
sns.scatterplot(x='PC1', y='PC2', hue='Species', data=ProjectedDataSklearnDfWithClassInfo)
plt.show()