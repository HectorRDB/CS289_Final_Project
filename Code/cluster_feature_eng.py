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
from sklearn.cluster import KMeans, MeanShift
import Test_Clustering

# Reading data
df = pd.read_csv('../cache/synthetic_data.csv', index_col=0)
df_dropped_points = pd.read_csv('../cache/dropped_points.csv', index_col=0)
points = df.values
dropped_points = df_dropped_points.values



def create_feature_matrix_per_gene(points, row, window_size,fill_missing):
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
                    
    return res_index,feature_matrix[:res_index,:], labels

"""
gene =7
s_test, features_test, labels_test = create_feature_matrix_per_gene(points,gene,10,"symmetry")


# PCA
print("Visualization after a 2-PCA: ")
pca = PCA(n_components=2)
pca.fit(features_test)


features_test_red = pca.transform(features_test)
plt.scatter(features_test_red[:,0],features_test_red[:,1],marker='.', c=labels_test, cmap=matplotlib.colors.ListedColormap(['red', 'green']))

"""

#unwanted genes (because unexpected NaN values)
unwanted=[365,965]


def create_feature_matrix_all_gene(points,window_size=10,fill_missing="symmetry"):
    n,m = points.shape
    features_matrix = np.zeros((n*m,window_size*2+1))
    all_labels = []
    total_size = 0
    for i in range(n):
        if i not in unwanted:
            s, features, labels = create_feature_matrix_per_gene(points,i,10,"symmetry")
            all_labels += labels
            
            features_matrix[total_size:total_size+s,:] = features
            
            total_size += s
    return total_size,features_matrix[:total_size],np.array(all_labels)
        
    

total_size,features,all_labels = create_feature_matrix_all_gene(points)


print("Visualization with the true labels after a 2-PCA: ")
num_genes = features.shape[0]
pca = PCA(n_components=2)
pca.fit(features[:num_genes])

print(pca.explained_variance_ratio_)  
print(pca.singular_values_) 

features_red = pca.transform(features[:num_genes])
plt.figure(figsize=(15,10))
plt.scatter(features_red[:,0],features_red[:,1],marker='.', c=all_labels[:num_genes], cmap=matplotlib.colors.ListedColormap(['red', 'green']))
plt.legend()
plt.show()


plt.figure(figsize=(30,20))
plt.scatter(features_red[:,0],features_red[:,1],marker='.', c=all_labels[:num_genes], cmap=matplotlib.colors.ListedColormap(['red', 'green']))
plt.legend()
plt.savefig("../figure/cluster_2_pca.png")
plt.close()



# Feature Engineering

min_expression =np.array([np.min(features[i,:]) for i in range(features.shape[0])]) #useless allways 0
max_expression =np.array([np.max(features[i,:]) for i in range(features.shape[0])]).reshape(-1,1)
num_zero_expression =np.array([(features[i,:]==0).shape[0] for i in range(features.shape[0])]).reshape(-1,1)

features_eng = np.copy(features)
features_eng = np.concatenate([features_eng,max_expression],axis=1)
features_eng = np.concatenate([features_eng,num_zero_expression],axis=1)



# PCA on Features Engineered
print("Visualization with the true labels after a 2-PCA and Feature Engineering: ")
num_genes = features_eng.shape[0]
pca = PCA(n_components=2)
pca.fit(features_eng[:num_genes])

print(pca.explained_variance_ratio_)  
print(pca.singular_values_) 

features_eng_red = pca.transform(features_eng[:num_genes])
plt.figure(figsize=(20,10))
plt.scatter(features_eng_red[:,0],features_eng_red[:,1],marker='.', c=all_labels[:num_genes], cmap=matplotlib.colors.ListedColormap(['red', 'green']))
plt.show()

plt.figure(figsize=(30,20))
plt.scatter(features_eng_red[:,0],features_eng_red[:,1],marker='.', c=all_labels[:num_genes], cmap=matplotlib.colors.ListedColormap(['red', 'green']))
plt.savefig("../figure/cluster_2_pca_with_fe.png")
plt.close()




# Clustering - Kmeans
kmeans = KMeans(n_clusters=2, random_state=0).fit(features_eng_red)
kmeans.labels_
kmeans.cluster_centers_

labels_predict = 1-kmeans.labels_


features_eng_red = pca.transform(features_eng[:num_genes])
print("Visualization with the predicted labels after a 2-PCA and Feature Engineering: ")
plt.figure(figsize=(20,10))
plt.scatter(features_eng_red[:,0],features_eng_red[:,1],marker='.', c=labels_predict[:num_genes], cmap=matplotlib.colors.ListedColormap(['red', 'green']))
plt.show()

plt.figure(figsize=(30,20))
plt.scatter(features_eng_red[:,0],features_eng_red[:,1],marker='.', c=labels_predict[:num_genes], cmap=matplotlib.colors.ListedColormap(['red', 'green']))
plt.savefig("../figure/k_means_2_pca_with_fe.png")
plt.close()



# Accuracy
labels_predict.shape
print("Accuracy: ",all_labels[all_labels==labels_predict].shape[0]/all_labels.shape[0])

TPR = all_labels[(all_labels==1.) & (all_labels ==labels_predict)].shape[0]/(all_labels[(all_labels==1.) & (all_labels ==labels_predict)].shape[0]+all_labels[(all_labels==1.) & (labels_predict ==0.)].shape[0])
print("TPR: ",TPR)
TNR = all_labels[(all_labels==0.) & (all_labels ==labels_predict)].shape[0]/(all_labels[(all_labels==0.) & (all_labels ==labels_predict)].shape[0]+all_labels[(all_labels==0.) & (labels_predict==1)].shape[0])
print("TNR: ",TNR)
acc= (all_labels[(all_labels==1.) & (all_labels ==labels_predict)].shape[0]+all_labels[(all_labels==0.) & (all_labels ==labels_predict)].shape[0])/(all_labels[(all_labels==1.) & (all_labels ==labels_predict)].shape[0]+all_labels[(all_labels==1.) & (labels_predict ==0.)].shape[0]+all_labels[(all_labels==0.) & (all_labels ==labels_predict)].shape[0]+all_labels[(all_labels==0.) & (labels_predict==1)].shape[0])
print("Accuracy : ",acc)




