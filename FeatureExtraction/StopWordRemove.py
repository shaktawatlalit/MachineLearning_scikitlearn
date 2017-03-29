# -*- coding: utf-8 -*-
"""
Created on Sat Mar 25 01:24:44 2017

@author: Lalit Singh
"""

from sklearn.feature_extraction.text import CountVectorizer
corpus = ['UNC played Duke in basketball', 'Duke lost the basketball game',
          'I ate a sandwich']
vectorizer = CountVectorizer(stop_words='english')
print(vectorizer.fit_transform(corpus).todense())
print(vectorizer.vocabulary_) 