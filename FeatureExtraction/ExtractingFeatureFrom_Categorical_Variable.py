# -*- coding: utf-8 -*-
"""
Created on Sat Mar 25 00:49:30 2017

@author: Lalit Singh
"""

from sklearn.feature_extraction import DictVectorizer
one_hot_encode = DictVectorizer()
instances = [{'city': 'New York'},  
              {'city': 'San Francisco'},
              {'city': 'Chapel Hill'}]  
print(one_hot_encode.fit_transform(instances).toarray())