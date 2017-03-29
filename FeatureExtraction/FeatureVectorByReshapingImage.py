# -*- coding: utf-8 -*-
"""
Created on Sat Mar 25 12:47:44 2017

@author: Lalit Singh
"""

from sklearn import datasets
digits = datasets.load_digits()
print("digit is :%s" %digits.target[0])
print("iamge is :%s" %digits.images[0])
""" Reshaping the matrix """
print("Feature vector is : \n", digits.images[0].reshape(-1,64))