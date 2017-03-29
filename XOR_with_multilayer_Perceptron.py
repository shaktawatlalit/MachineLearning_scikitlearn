# -*- coding: utf-8 -*-
"""
Created on Tue Mar 28 19:24:24 2017

@author: Lalit Singh
"""

from sklearn.cross_validation import train_test_split
from sklearn.neural_network import MultilayerPerceptronClassifier
y = [0, 1, 1, 0] * 1000
X = [[0, 0], [0, 1], [1, 0], [1, 1]] * 1000
X_train, X_test, y_train, y_test = train_test_split(X, y,random_state=3) 
clf = MultilayerPerceptronClassifier(n_hidden=[2],activation='logistic',algorithm='sgd', random_state=3)
clf.fit(X_train, y_train)
print('Number of layers: %s. Number of outputs: %s' % (clf.n_ layers_, clf.n_outputs_))
predictions = clf.predict(X_test)
print('Accuracy:', clf.score(X_test, y_test))
for i, p in enumerate(predictions[:10]):
    print('True: %s, Predicted: %s' % (y_test[i], p))
