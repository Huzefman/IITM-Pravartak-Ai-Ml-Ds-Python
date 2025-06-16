#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 16 14:12:55 2025

@author: huzefa
"""

import numpy as np

#Creating Array
my_list=[1,2,3,4,5,6]
print(my_list)
array=np.array(my_list, dtype=int)
print(array)

#Properties of array
print(type(array))
print(len(array))
print(array.ndim)
print(array.shape)

#Reshaping array
array2=array.reshape(3, 2)
print(array2)
array2.shape

array3=array.reshape(3,-1)
print(array3)
print(array3.ndim)

#Initializing numpy arrays from Python lists
my_list2=[1,2,3,4,5]
my_list3=[2,2,3,4,5]
my_list4=[3,2,3,4,5]
mul_arr=np.array([my_list2,my_list3,my_list4])
print(mul_arr)

mul_arr.reshape(1,15)

#NumPy Attributes
a=np.array([[1,2,3],[4,5,6]])
print(a.shape)

#Reshaping the ndarray
a.shape=(3,2)
print(a)

#Reshape function to resize an array
b=a.reshape(3,2)
print(b)

r=range(24)
print(r)

#Array of evenly spaced numbers
a=np.arange(24)
print(a)
print(a.ndim)

#Reshaping the array 'a'
b=a.reshape(6,4,1)
print(b)

#Length of each element in array
x=np.array([1,2,3,4,5], dtype=np.int8)
print(x.itemsize)
x=np.array([1,2,3,4,5], dtype=np.float32)
print(x.itemsize)

#Arithmetic Operations
x=np.array([[1,2],[3,4]])
y=np.array([[5,6],[7,8]])
print(x)
print(y)

#Addition
print(x+y)
print(np.add(x,y))

#Subtraction
print(x-y)
print(np.subtract(x,y))

#Multiplication
print(x*y)
print(np.multiply(x,y))

#Dot
print(x.dot(y))
print(np.dot(x,y))

#Divide
print(x/y)
print(np.divide(x,y))

#Row, Column wise
print(np.sum(x))
print(np.sum(x, axis=0))
print(np.sum(x, axis=1))
