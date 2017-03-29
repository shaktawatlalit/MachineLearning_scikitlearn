# -*- coding: utf-8 -*-
"""
Created on Fri Mar 24 16:22:11 2017

@author: Lalit Singh
"""
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import numpy as np
X = [[6], [8], [10], [14],   [18]] 
y = [[7], [9], [13], [17.5], [18]] 

""" ploting graph of data """

plt.figure() 
plt.title('Pizza price plotted against diameter')
plt.xlabel('Diameter in inches') 
plt.ylabel('Price in dollars') 
plt.plot(X, y, 'k.')
plt.axis([0, 25, 0, 25])
plt.grid(True) 
plt.show() 

""" fitting linear regression model """
model=LinearRegression()
model.fit(X,y)
print("price of 12 inch pizza is :$%.2f" %model.predict([18])[0])

""" Residual sum of square """

print("residual sum of square is :$%.2f" %np.mean((model.predict(X)-y)**2))

""" calculate variance"""

print("variance is :$%.2f" %np.var(X ,ddof=1))

""" for computing r-squared error"""
X_test = [[8],  [9],   [11], [16], [12]] 
y_test = [[11], [8.5], [15], [18], [11]]
print("r squared error is :$%.2f" %model.score(X_test,y_test)) 