# -*- coding: utf-8 -*-
"""
Created on Fri Mar 24 19:36:15 2017

@author: Lalit Singh
"""

from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cross_validation import train_test_split,cross_val_score
df = pd.read_csv('winequality-red.csv', sep=';')
X=df[list(df.columns)[:-1]]
y=df['quality']
x_train,X_test,y_train,y_test = train_test_split(X,y)
regression = LinearRegression()
regression.fit(X,y)
y_predict = regression.predict(X_test)
print("r squared score is : $%.2f" %regression.score(X_test,y_test))


""" cross_validation"""
model = LinearRegression()
score = cross_val_score(model,X,y,cv=6)
print(score.mean())
print(score)