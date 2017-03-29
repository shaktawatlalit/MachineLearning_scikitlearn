# -*- coding: utf-8 -*-
"""
Created on Tue Mar 28 19:38:01 2017

@author: Lalit Singh
"""

from sklearn.datasets import load_digits 
from sklearn.cross_validation import train_test_split, cross_val_score 
from sklearn.pipeline import Pipeline 
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network.multilayer_perceptron import MultilayerPerceptronClassifier
if __name__ == '__main__':
    digits = load_digits() 
    X = digits.data
    y = digits.target
    
pipeline = Pipeline([('ss', StandardScaler()),('mlp', MultilayerPerceptronClassifier(n_hidden=[150, 100], alpha=0.1))]) 
print(cross_val_score(pipeline, X, y, n_jobs=-1)
    
