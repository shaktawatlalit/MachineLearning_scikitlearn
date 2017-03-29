# -*- coding: utf-8 -*-
"""
Created on Sun Mar 26 12:10:43 2017

@author: Lalit Singh
"""

import numpy as np
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt 
cluster1 = np.random.uniform(0.5, 1.5, (2, 10))
cluster2 = np.random.uniform(3.5, 4.5, (2, 10))
X = np.hstack((cluster1, cluster2)).T 
X = np.vstack((x, y)).T 
K = range(1, 10)
meandistortions = [] 
   for k in K:
       kmeans = KMeans(n_clusters=k) 
       kmeans.fit(X)
       meandistortions.append(sum(np.min(cdist(X, kmeans.cluster_ centers_, 'euclidean'), axis=1)) / X.shape[0]) 
plt.plot(K, meandistortions, 'bx-')
plt.xlabel('k')
plt.ylabel('Average distortion')
plt.title('Selecting k with the Elbow Method')
plt.show()
