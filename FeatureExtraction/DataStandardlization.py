# -*- coding: utf-8 -*-
"""
Created on Sat Mar 25 13:28:58 2017

@author: Lalit Singh
"""

from sklearn import preprocessing
import numpy as np
X = np.array([ [0., 0., 5., 13., 9., 1.],
               [0., 0., 13., 15., 10., 15.],
               [0., 3., 15., 2., 0., 11.] ]) 
print(preprocessing.scale(X))