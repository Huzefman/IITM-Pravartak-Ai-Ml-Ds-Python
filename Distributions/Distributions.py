#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 17 18:38:04 2025

@author: huzefa
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats

#Setting the default figure size
plt.rcParams["figure.figsize"]=(7,7)

#Generating random numbers (sample)
v1=scipy.stats.norm.rvs(loc=0,scale=1,size=10)
print('Mean',v1.mean())
print('Standard Deviation',v1.std())

#Visualising the sample
sns.displot(v1)
plt.show()

#Repeating with larger sample sizes
v1=scipy.stats.norm.rvs(loc=0,scale=1,size=100)
print('Mean',v1.mean())
print('Standard Deviation',v1.std())

#Visualising the sample
sns.displot(v1)
plt.show()

v1=scipy.stats.norm.rvs(loc=0,scale=1,size=1000)
print('Mean',v1.mean())
print('Standard Deviation',v1.std())

#Visualising the sample
sns.displot(v1)
plt.show()

#Probability density function pdf
scipy.stats.norm.pdf(-1)
scipy.stats.norm.pdf(np.arange(-3,-1,0.01 ),loc=0,scale=1)

#Cumulative distribution function cdf
scipy.stats.norm.cdf(x=-1,loc=0,scale=1) #for -inf to -1
1-scipy.stats.norm.cdf(x=-1,loc=0,scale=1) #for -1 to inf

#Visualisation of distribution
plt.fill_between(x=np.arange(-3,3,0.01), y1=scipy.stats.norm.pdf(np.arange(-3,3,0.01)),facecolor='blue',alpha=0.5)
plt.vlines(x=-1, ymin=0, ymax=.24, linestyles='dashed')
plt.show()

#Plotting with cdf value on the distribution
#Left side
prob_under_neg1=scipy.stats.norm.cdf(x=-1,loc=0,scale=1)
plt.fill_between(x=np.arange(-3,-1,0.01), y1=scipy.stats.norm.pdf(np.arange(-3,-1,0.01)), facecolor='blue',edgecolor='black',alpha=0.5)
plt.fill_between(x=np.arange(-1,3,0.01), y1=scipy.stats.norm.pdf(np.arange(-1,3,0.01)), facecolor='orange',edgecolor='black',alpha=0.5)
plt.text(x=-2, y=.03, s=round(prob_under_neg1,3))
plt.vlines(x=-1, ymin=0, ymax=.24, linestyles='dashed')
plt.show()

#Right Side
prob_under_neg1=1-scipy.stats.norm.cdf(x=-1,loc=0,scale=1)
plt.fill_between(x=np.arange(-3,-1,0.01), y1=scipy.stats.norm.pdf(np.arange(-3,-1,0.01)), facecolor='blue',edgecolor='black',alpha=0.5)
plt.fill_between(x=np.arange(-1,3,0.01), y1=scipy.stats.norm.pdf(np.arange(-1,3,0.01)), facecolor='orange',edgecolor='black',alpha=0.5)
plt.text(x=1, y=.03, s=round(prob_under_neg1,3))
plt.vlines(x=-1, ymin=0, ymax=.24, linestyles='dashed')
plt.show()

#For bounded plot
prob_under_neg1=scipy.stats.norm.cdf(x=-1,loc=0,scale=1)
prob_over_pos1=1-prob_under_neg1
between_prob=1-(prob_over_pos1+prob_under_neg1)
plt.fill_between(x=np.arange(-3,-1,0.01), y1=scipy.stats.norm.pdf(np.arange(-3,-1,0.01)), facecolor='blue',edgecolor='black',alpha=0.5)
plt.fill_between(x=np.arange(-1,1,0.01), y1=scipy.stats.norm.pdf(np.arange(-1,1,0.01)), facecolor='orange',edgecolor='black',alpha=0.5)
plt.fill_between(x=np.arange(1,3,0.01), y1=scipy.stats.norm.pdf(np.arange(1,3,0.01)), facecolor='blue',edgecolor='black',alpha=0.5)
plt.show()

#For tails
tails_prob=prob_under_neg1+prob_over_pos1

#Inverse cdf
q_val=scipy.stats.norm.cdf(x=-1,loc=0,scale=1)
scipy.stats.norm.ppf(q=q_val,loc=0,scale=1)

#Binomial distribution
scipy.stats.binom.rvs(n=10,p=0.5)
scipy.stats.binom.rvs(size=5,n=10,p=.05,random_state=0)

#Visualisation of binomial distribution
data = scipy.stats.binom.rvs(n=10, p=0.5, size=1000, random_state=0)
plt.hist(data, bins=range(12), color='b', alpha=0.8, edgecolor='black')
plt.xlabel('Number of successes')
plt.ylabel('Frequency of success in trials')
plt.title('Binomial Distribution')
plt.show()

#Probability mass function pmf
scipy.stats.binom.pmf(n=20,p=0.5,k=9)

k_range=np.arange(0,10)
scipy.stats.binom.pmf(n=20,p=0.5,k=k_range)
sum(scipy.stats.binom.pmf(n=20,p=0.5,k=k_range))
scipy.stats.binom.cdf(n=20,p=0.5,k=9)