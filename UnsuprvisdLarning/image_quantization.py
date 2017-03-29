# -*- coding: utf-8 -*-
"""
Created on Sun Mar 26 14:40:06 2017

@author: Lalit Singh
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.utils import shuffle
import mahotas as mh 
"""First we read and flatten the image:"""
original_img = np.array(mh.imread('sexy.jpg'), dtype=np. float64) / 255
original_dimensions = tuple(original_img.shape)
width, height, depth = tuple(original_img.shape)
image_flattened = np.reshape(original_img, (width * height, depth))

""" We then use K-Means to create 64 clusters from a sample of 1,000 randomly selected colors. Each of the clusters will be a color in the compressed palette."""

image_array_sample = shuffle(image_flattened, random_state=0) [:1000]
estimator = KMeans(n_clusters=64, random_state=0)
estimator.fit(image_array_sample) 
 
""" we predict the cluster assignment for each of the pixels in the original image:"""
cluster_assignments = estimator.predict(image_flattened)

"""Finally, we create the compressed image from the compressed palette and  cluster assignments:"""
compressed_palette = estimator.cluster_centers_ 
compressed_img = np.zeros((width, height, compressed_palette.shape[1]))
label_idx = 0
for i in range(width):
    for j in range(height):
        compressed_img[i][j] = compressed_palette[cluster_assignments[label_idx]]
        label_idx += 1
plt.subplot(122)
plt.title('Original Image')
plt.imshow(original_img)
plt.axis('off')
plt.subplot(121)
plt.title('Compressed Image')
plt.imshow(compressed_img)
plt.axis('off')
plt.show() 