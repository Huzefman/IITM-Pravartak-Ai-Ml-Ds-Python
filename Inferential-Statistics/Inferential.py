#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 18 14:02:25 2025

@author: huzefa
"""

import pandas as pd
import scipy.stats

#Importing the dataset
cars_data=pd.read_csv('../Datasets/clean_cars_sampled.csv')

#Creating a copy of dataset
cars=cars_data.copy()

#Setting working range
cars=cars[(cars['yearOfRegistration']<=2018) & (cars['yearOfRegistration']>=1950) & (cars['price']>=100) & (cars['price']<=150000) & (cars['powerPS']>=10) & (cars['powerPS']<=500)]

#One sample test for mean (has price changed from $6000 since last 3 years)
#Taking a sample of 1000 and alpha=0.05
sample_size=1000
sample1=cars.sample(sample_size,random_state=0)

#Postulated mean
pos_mean=6000
#Sample calculated mean
print(sample1['price'].mean())

#Getting test staistic value and p value
from scipy.stats import ttest_1samp
statistic,pvalue=ttest_1samp(sample1['price'], pos_mean)

#Calculating critical values
#Getting degrees of freedom
n=len(cars['price'])
df=n-1
print(n,df)
alpha=0.05
#t distribution
from scipy.stats import t
cv=t.ppf([alpha/2,1-alpha/2],df)
print('Critical Values:',cv)
print('Test statistic:',statistic)
print('pvalue:',pvalue)
print('Do not reject null hypotheisis as test statistic lie in critical range and pvalue is greater than alpha (0.05)')

#One sample test for proportion (has automatic transmission changed from 23% since last 3 years)
from statsmodels.stats.proportion import proportions_ztest
p0=0.23

#Count of automatic transmission
count=sample1['gearbox'].value_counts()[1]
#Proportion of different transmissions
nobs=len(sample1['gearbox'])
sample1['gearbox'].value_counts()/nobs

#Calculating test statistic and pvalue
statistic_oneprop,pvalue_oneprop=proportions_ztest(count=count, nobs=nobs, value=p0, alternative='two-sided', prop_var=False)
print(statistic_oneprop, pvalue_oneprop)

#Getting critical values from normal distribution
from scipy.stats import norm
cv_norm=norm.ppf([alpha/2,1-alpha/2])
print(cv_norm)
print('Do not reject null hypotheisis as test statistic lie in critical range and pvalue is greater than alpha (0.05)')

#Two sample test for mean (is the mean price for 30k-60k KM same as 70k-90k KM)
#Subsetting the data
km_70_90=cars[(cars.kilometer <= 90000) & (cars.kilometer>=70000)]
km_30_60=cars[(cars.kilometer <= 60000) & (cars.kilometer>=30000)]
sample_70_90_km=km_70_90.sample(500,random_state=0)
sample_30_60_km=km_30_60.sample(500,random_state=0)

#Sample Variance
print(sample_30_60_km.price.var())
print(sample_70_90_km.price.var())
#Sample Mean
print(sample_30_60_km.price.mean())
print(sample_70_90_km.price.mean())

#Computing f statistic
from scipy.stats import f
F=sample_70_90_km.price.var()/sample_30_60_km.price.var()
print(F)

#Calculating degrees of freedom
df1=len(sample_30_60_km)-1
df2=len(sample_70_90_km)-1

#Getting true f value
fvalue=scipy.stats.f.cdf(F, df1, df2)
print(fvalue)

#Critical values
f.ppf([alpha/2,1-alpha/2],df1,df2)
print('Reject null hypotheisis as test statistic do not lie in critical range and fvalue is less than alpha (0.05)')

#Welch t test for unqeual variances
from scipy.stats import ttest_ind
statistic_twomean, pvalue_twomean=ttest_ind(sample_30_60_km.price, sample_70_90_km.price,equal_var=False)
print(statistic_twomean,pvalue_twomean)

#Calculating degree of freedom
N1=len(sample_30_60_km)
N2=len(sample_70_90_km)
s12=sample_30_60_km.price.var()
s22=sample_70_90_km.price.var()
df=(((s12/N1)+(s22/N2))**2)/((((s12/N1)**2)/(N1-1))+(((s22/N2)**2)/(N2-1)))
print(df)

#Critical values
cv_t=t.ppf([alpha/2,1-alpha/2],df)
print(cv_t)
print('Reject null hypotheisis as test statistic do not lie in critical range and pvalue is less than alpha (0.05)')

#Two sample test for proportion (are proportion of petrol cars from 2009-2013 and 2014-2018 different)
#Subsetting based on year
year_14_18=cars[(cars.yearOfRegistration<=2018) & (cars.yearOfRegistration>=2014)]
year_09_13=cars[(cars.yearOfRegistration<=2013) & (cars.yearOfRegistration>=2009)]

#Taking 1000 samples
sample_14_18=year_14_18.sample(1000,random_state=3)
sample_09_13=year_09_13.sample(1000,random_state=3)

#Calculating the proportion of both
from statsmodels.stats.proportion import proportions_ztest
count=[(sample_14_18['fuelType']=='petrol').sum(),(sample_09_13['fuelType']=='petrol').sum()]
nobs=[len(sample_14_18),len(sample_09_13)]
print(count[0]/nobs[0])
print(count[1]/nobs[1])

#Calculating statistic and pvalue
statistic,pvalue=proportions_ztest(count=count,nobs=nobs,value=0,alternative='two-sided',prop_var=False)
print(statistic,pvalue)

#Getting normal critical values
cv=norm.ppf([alpha/2,1-alpha/2])
print(cv)
print('Reject null hypotheisis as test statistic do not lie in critical range and pvalue is less than alpha (0.05)')

#Chi-square test of independence (is vehicleType dependent on fuelType)
#Setting crosstab between fueltype and vehicletype
cross_table=pd.crosstab((cars['fuelType']),cars['vehicleType'])

#Applying function chi2_contigency
cont=scipy.stats.chi2_contingency(cross_table)
print(cont)

#Calculating degrees of freedom
df=(cross_table.shape[0]-1)*(cross_table.shape[1]-1)
print(df)

#Getting critical values
from scipy.stats import chi2
chi2.ppf(q=[alpha/2,1-alpha/2],df=42)
print('Reject null hypotheisis as chi2 statistic do not lie in critical range and pvalue is less than alpha (0.05) conclude vehicleType is not dependent on fuelType')













