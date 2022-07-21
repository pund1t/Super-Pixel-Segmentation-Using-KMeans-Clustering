#!/usr/bin/env python
# coding: utf-8

# In[5]:


import numpy as np
import random
from scipy.spatial.distance import cdist
from scipy.spatial import distance_matrix

class kmeans:
    
    def __init__(self , k):
        
        self.k = k     
        self.centroids = None
        self.X = None
        self.clusters = None
        
    def getCentorids(self):
        
        return self.centroids
    
    def getClusters(self):
        
        return self.clusters
    
    def fit(self , X , max_iter = 200):
        
        npt , ndm = X.shape
        
        self.centroids = np.random.uniform(low = np.min(X) , high = np.max(X) , size = (self.k,ndm))
        
        for i in range(max_iter):
            
            diff = cdist(X, self.centroids, metric="euclidean")
            self.clusters = np.argmin(diff, axis=1)

            for i in range(self.k):

                self.centroids[i] = np.mean(X[np.where(self.clusters == i)] , axis=0)
                
    def predict(self , X):
        
        diff = cdist(X , self.centroids , metric="euclidean")
        return np.argmin(diff , axis=1)
        
        
        
        
        


# In[ ]:




