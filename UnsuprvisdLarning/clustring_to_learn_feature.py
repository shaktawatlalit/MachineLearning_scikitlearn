# -*- coding: utf-8 -*-
"""
Created on Sun Mar 26 15:01:19 2017

@author: Lalit Singh
"""

import numpy as np
import mahotas as mh
from mahotas.features import surf
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import *
from sklearn.cluster import MiniBatchKMeans
import glob

""" First, we load the images, convert them to grayscale, and extract the SURF descriptors."""
all_instance_filenames = []
all_instance_targets = []
for f in glob.glob('cats-and-dogs-img/*.jpg'):
     target = 1 if 'cat' in f else 0 
     all_instance_filenames.append(f)
     all_instance_targets.append(target)
surf_features = []
counter = 0
for f in all_instance_filenames:
    print('Reading image:', f)
    image = mh.imread(f, as_grey=True)
    surf_features.append(surf.surf(image)[:, 5:])
train_len = int(len(all_instance_filenames) * .60)
X_train_surf_features = np.concatenate(surf_features[:train_len])
X_test_surf_feautres = np.concatenate(surf_features[train_len:])
y_train = all_instance_targets[:train_len]
y_test = all_instance_targets[train_len:]

""" We then group the extracted descriptors into 300 clusters in the following code sample. We use MiniBatchKMeans, a variation of K-Means that uses a random sample of the instances in each iteration. As it computes the distances to the centroids for only a sample of the instances in each iteration"""
n_clusters = 300
print('Clustering', len(X_train_surf_features), 'features')
estimator = MiniBatchKMeans(n_clusters=n_clusters)
estimator.fit_transform(X_train_surf_features)
X_train = []
for instance in surf_features[:train_len]:
    clusters = estimator.predict(instance)
    features = np.bincount(clusters)
    if len(features) < n_clusters:
        features = np.append(features, np.zeros((1, n_clusterslen(features))))
    X_train.append(features)
X_test = []
for instance in surf_features[train_len:]:
    clusters = estimator.predict(instance)
    features = np.bincount(clusters)
    if len(features) < n_clusters:
        features = np.append(features, np.zeros((1, n_clusterslen(features))))
    X_test.append(features) 
    
    
 """Finally, we train a logistic regression classifier on the feature vectors and targets,  and assess its precision, recall, and accuracy:"""
clf = LogisticRegression(C=0.001, penalty='l2')
clf.fit_transform(X_train, y_train)
predictions = clf.predict(X_test)
print classification_report(y_test, predictions)
print('Precision: ', precision_score(y_test, predictions))
print 'Recall: ', recall_score(y_test, predictions)
print 'Accuracy: ', accuracy_score(y_test, predictions)


