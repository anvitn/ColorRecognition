"""FinalProject-Color.ipynb
"""

## Importing the Necessary Libraries
import warnings
import cv2
import numpy as np
import os
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

## Importing an Image

image = cv2.imread("/content/Color-yellow.jpg")
print(image.shape)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

## Showing an Image

plt.imshow(image)
plt.show()

## Reshaping the Image

X = image.reshape((-1,3))
print(X.shape)

## KMeans Algorithm

k = 5
model = KMeans(n_clusters=k, random_state=42)
model.fit(X)

## Getting centroids of clusters

centroids = model.cluster_centers_
print(centroids)

## Printing Colors

colors = np.array(centroids, dtype='uint8')
print(colors)

## Color Patches for dominant colors

i = 1

for color in colors:
  plt.subplot(1,k,i)
  i = i+1
  patch = np.zeros((100,100,3), dtype='uint8')
  patch[:,:,:] = color
  plt.imshow(patch)
  plt.axis('off')
plt.show()