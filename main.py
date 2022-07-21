#!/usr/bin/env python
# coding: utf-8




import numpy as np
import matplotlib.pyplot as plt
from kmeans import kmeans
import cv2

plt.rcParams.update({'font.size': 22})


image = cv2.imread("images/input/image.jpg")
image = cv2.cvtColor(image , cv2.COLOR_BGR2RGB)
plt.figure(figsize = (8,6))
plt.imshow(image)
plt.xlabel('original image')
plt.tick_params(labelleft=False, labelbottom=False, labelright=False, labeltop=False)
plt.show()

X = np.float32(image.reshape((-1,3)))

kgif = [1,2,5,10,25,50]

for k in kgif:
    km = kmeans(k = k)
    km.fit(X)
    centers = km.getCentorids()
    clusters = km.getClusters()
    segimg = centers[clusters]
    segimg = segimg.reshape(image.shape)
    
    plt.figure(figsize = (8,6))
    plt.imshow((segimg).astype(np.uint8))
    plt.xlabel('k = ' + str(k))
    plt.tick_params(labelleft=False, labelbottom=False, labelright=False, labeltop=False)
    plt.show()







