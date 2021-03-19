'''
Authors - Andrew Vo & Anant Natekar
'''
import nltk
import os
import re
import random  
import string
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn import metrics
from sklearn import datasets
from sklearn.cluster import KMeans, AgglomerativeClustering

#path = "/Users/nat2147/Anant/code/python/" #defining the path to the training data files
#feature_vectors = load_svmlight_file(path + "trainingdatafile.txt")
iris = datasets.load_iris()
feature_vectors = iris['data']
classification_labels = iris['target']

#kmeans clustering
sc = []
nmi = []
for number_of_clusters in range (2, 26):
    km = KMeans(number_of_clusters).fit(feature_vectors)
    clustering_labels = km.labels_
    sc.append(metrics.silhouette_score(feature_vectors, clustering_labels, metric='euclidean'))
    nmi.append(metrics.normalized_mutual_info_score(classification_labels, clustering_labels))
    
plt.title("K Means Clustering", fontsize=15)
plt.plot(sc, color='b', linestyle='-', linewidth=2, label = "Sihouette Coefficient")
plt.plot(nmi, color='g', linestyle='-', linewidth=2, label = "Normalized Mutual Information")
plt.xlabel("Number of Clusters",fontsize=13)
plt.ylabel("Measures",fontsize=13)
plt.legend(loc = 1)
#plt.show(block=False)
plt.show()

#hierarchical (Agglomerative) clustering
for number_of_clusters in range (2, 26):
    single_linkage_model = AgglomerativeClustering(
        n_clusters=number_of_clusters, linkage='ward').fit(feature_vectors)
    clustering_labels = single_linkage_model.labels_
    sc.append(metrics.silhouette_score(feature_vectors, clustering_labels, metric='euclidean'))
    nmi.append(metrics.normalized_mutual_info_score(classification_labels, clustering_labels))
    
plt.title("Hierarchical Clustering", fontsize=15)
plt.plot(sc, color='b', linestyle='-', linewidth=2, label = "Sihouette Coefficient")
plt.plot(nmi, color='g', linestyle='-', linewidth=2, label = "Normalized Mutual Information")
plt.xlabel("Number of Clusters",fontsize=13)
plt.ylabel("Measures",fontsize=13)
plt.legend(loc = 1)
#plt.show(block=False)
plt.show()