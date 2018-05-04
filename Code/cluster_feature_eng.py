#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May  4 11:19:23 2018

@author: daniel
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


df = pd.read_csv('../cache/synthetic_data.csv', index_col=0)



plt.scatter(np.arange(500).reshape(1,500),df.iloc[[0]],)



points = df.values

def create_feature_matrix_per_row(points, row, window_size=10):
    n,m = points.shape
    feature_matrix = np.zeros((m,window_size*2+1))
    for j in range(m):
        for k in range(-window_size,window_size+1):
            if j+k<0:
                feature_matrix[j,k+window_size] = 'NaN'
            elif j+k>m-1:
                feature_matrix[j,k+window_size] = 'NaN'
            else: 
                feature_matrix[j,k+window_size] = points[row,j+k]
                
    
            


    return (feature_matrix)

def create_feature_matrix(points,window_size=10):
    n,m = points.shape
    features_matrix = np.zeros((n*m,window_size*2+1))
    for i in range(n):
        features_matrix[i*m:i*m+m,:] = create_feature_matrix_per_row(points, i, window_size)
    
    return features_matrix
        
    

features = create_feature_matrix(points, 10)
