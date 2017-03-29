# -*- coding: utf-8 -*-
"""
Created on Sat Mar 25 01:06:55 2017

@author: Lalit Singh
"""

from sklearn.feature_extraction.text import CountVectorizer
corpus = ['UNC played Duke in basketball', 'Duke lost the basketball game']
count_vectorizer = CountVectorizer()
print(count_vectorizer.fit_transform(corpus).todense())
print(count_vectorizer.vocabulary_)  