#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May  4 11:19:23 2018

@author: daniel
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
from sklearn.decomposition import PCA

# Reading data
df = pd.read_csv('../cache/synthetic_data.csv', index_col=0)
df_dropped_points = pd.read_csv('../cache/dropped_points.csv', index_col=0)
points = df.values
dropped_points = df_dropped_points.values



def create_feature_matrix_per_row(points, row, window_size,fill_missing):
    """
    fill_missing = "symmetry" or "NaN" to chose the option to complet the missing value.
    """
    print("Processing of gene number: ",row)
    n,m = points.shape
    feature_matrix = np.zeros((m,window_size*2+1))
    
    labels = []
    
    res_index = 0
    for j in range(m):
        if points[row,j] ==0.:
            res_index +=1
            labels.append(dropped_points[row,j])
            
            for k in range(-window_size,window_size+1):
                if j+k<0:
                    if fill_missing=="NaN":
                        feature_matrix[res_index,k+window_size] = 'NaN'
                    elif fill_missing=="symmetry":
                        feature_matrix[res_index,k+window_size] = points[row,j-k]
                elif j+k>m-1:
                    if fill_missing=="NaN":
                        feature_matrix[res_index,k+window_size] = 'NaN'
                    elif fill_missing=="symmetry":
                        feature_matrix[res_index,k+window_size] = points[row,j-k]
                else: 
                    feature_matrix[res_index,k+window_size] = points[row,j+k]
                    
    return res_index,feature_matrix[:res_index,:], np.array(labels)


gene =2
s_test, features_test, labels_test = create_feature_matrix_per_row(points,gene,10,"symmetry")


# PCA
print("Visualization after a 2-PCA: ")
pca = PCA(n_components=2)
pca.fit(features_test)


features_test_red = pca.transform(features_test)
plt.scatter(features_test_red[:,0],features_test_red[:,1],marker='.', c=labels_test, cmap=matplotlib.colors.ListedColormap(['red', 'green']))





"""
def create_feature_matrix(points,window_size=10,fill_missing="symmetry"):
    n,m = points.shape
    features_matrix = np.zeros((n*m,window_size*2+1))
    for i in range(n):
        features_matrix[i*m:i*m+m,:] = create_feature_matrix_per_row(points, i, window_size,fill_missing)
    
    return features_matrix
        
    

features = create_feature_matrix(points)
"""

