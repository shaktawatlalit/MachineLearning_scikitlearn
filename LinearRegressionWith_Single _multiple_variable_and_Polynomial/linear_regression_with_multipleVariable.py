# -*- coding: utf-8 -*-
"""
Created on Fri Mar 24 17:06:51 2017

@author: Lalit Singh
"""

from sklearn.linear_model import LinearRegression
import numpy as np
X = [[6, 2], [8, 1], [10, 0], [14, 2], [18, 0]] 
y = [[7],    [9],    [13],    [17.5],  [18]] 
model=LinearRegression()
model.fit(X,y)
X_test = [[8, 2], [9, 0], [11, 2], [16, 2], [12, 0]] 
y_test = [[11],   [8.5],  [15],    [18],    [11]] 

""" calculate r-squared error """
print("r-squared error is :$%.2f"%model.score(X_test,y_test))

""" predicted value against test_data"""
predictions = model.predict(X_test)
for i, predict in enumerate(predictions):
    print("predicted data is :%s agianst the observed data :%s" %(predict,y_test[i]))