# -*- coding: utf-8 -*-
"""
Created on Fri Mar 24 18:52:39 2017

@author: Lalit Singh
"""

from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
X_train = [[6], [8], [10], [14],   [18]] 
y_train = [[7], [9], [13], [17.5], [18]]
X_test = [[6],  [8],   [11], [16]]
y_test = [[8], [12], [15], [18]]

""" fit linear transformation """
regressor = LinearRegression()
regressor.fit(X_train,y_train)
xx = np.linspace(0,26,100)
yy = regressor.predict(xx.reshape(xx.shape[0], 1)) 
plt.plot(xx,yy)


""" quardetic """
quardetic = PolynomialFeatures(degree=2)
X_train_quardetic = quardetic.fit_transform(X_train)
X_test_quardetic = quardetic.transform(X_test)


regression_quardetic = LinearRegression()
regression_quardetic.fit(X_train_quardetic,y_train)
xx_quardetic = quardetic.transform(xx.reshape(xx. shape[0], 1))
yy_quardetic = regression_quardetic.predict(xx_quardetic)
plt.plot(xx_quardetic ,yy_quardetic, c='r', linestyle='--' )

""" plotting of original data """
plt.title('Pizza price regressed on diameter')
plt.xlabel('Diameter in inches')
plt.ylabel('Price in dollars')
plt.axis([0, 25, 0, 25])
plt.grid(True) 
plt.scatter(X_train, y_train)
plt.show()

""" r-squared """
print(" r squared of linear model is :$%.2f" %regressor.score(X_test,y_test))
print("r squared of quardetic model is :$%.2f" %regression_quardetic.score(X_test_quardetic,y_test))


