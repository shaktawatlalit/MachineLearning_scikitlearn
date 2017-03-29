# -*- coding: utf-8 -*-
"""
Created on Sat Mar 25 15:36:38 2017

@author: Lalit Singh
"""

# -*- coding: utf-8 -*-
"""
Created on Sat Mar 25 14:08:54 2017

@author: Lalit Singh
"""

import pandas as pd
from sklearn.metrics import f1_score
from sklearn.preprocessing import LabelBinarizer
import numpy as np
from sklearn.linear_model.logistic import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cross_validation import train_test_split, cross_val_score
df=pd.read_csv('SMSSpamCollection',delimiter='\t',header=None)
""" count of spam and ham messages
print(df.head())
print("number of spam messages : %d" % df[df[0]=="spam"][0].count())
print("number of ham messages : %d" %df[df[0]=="ham"][0].count() )  """

""" split the data into train test data """
X_train_raw, X_test_raw, y_train, y_test = train_test_split(df[1],df[0])

"""  Feature Extraction  """
vectorizer = TfidfVectorizer()
X_train = vectorizer.fit_transform(X_train_raw)
X_test  = vectorizer.transform(X_test_raw)

""" creating logistic model and fitting data """

classifier = LogisticRegression()
classifier.fit(X_train,y_train)
prediction = classifier.predict(X_test)

for i, predict in enumerate(prediction[:5]):
  print("prediction is  : %s " %predict)
  print("mesasage was : %s" %X_test[i])
  
  """ checking cross_validation_score """
score = cross_val_score(classifier,X_train,y_train,cv=5)
print(np.mean(score))
print(score)

""" calculating Precision  """
lb = LabelBinarizer()
y_train = np.array([number[0] for number in lb.fit_transform(y_train)])
precision_score = cross_val_score(classifier, X_train, y_train, cv=5, scoring='precision')
print("precision mean is : $%.2f " %np.mean(precision_score))
print("precision is : %s " %precision_score)

""" calculating Recall """
recall_score = cross_val_score(classifier, X_train, y_train, cv=5, scoring='recall')
print("recall mean is : $%.2f " %np.mean(recall_score))
print("recall is : %s " %recall_score)


""" calculating the f1 score """
f1_score = cross_val_score(classifier, X_train, y_train, cv=5, scoring='f1')
print("f1_score mean is : $%.2f " %np.mean(f1_score))
print("f1_score is : %s " %f1_score)


