# -*- coding: utf-8 -*-
"""
Created on Sat Mar 25 14:58:43 2017

@author: Lalit Singh
"""

from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
y_test = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]
y_pred = [0, 1, 0, 0, 0, 0, 0, 1, 1, 1] 
confusion_mat = confusion_matrix(y_test,y_pred)
print(confusion_mat)
plt.matshow(confusion_mat)
plt.title('Confusion matrix')
plt.colorbar()
plt.ylabel('True label') 
plt.xlabel('Predicted label') 
plt.show()
 
""" calculating accuracy score """
print(" accuracy score of logistic regression is :$%.2f" % accuracy_score(y_test,y_pred) )