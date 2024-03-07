#!/usr/bin/env python
# coding: utf-8


# Native libraries
import os
import math
# Essential Libraries
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Preprocessing
from sklearn.preprocessing import MinMaxScaler
# Algorithms
from minisom import MiniSom
from tslearn.barycenters import dtw_barycenter_averaging
from tslearn.clustering import TimeSeriesKMeans
from sklearn.cluster import KMeans

from sklearn.decomposition import PCA

import mat73

import scipy.io as spio

import sktime

#Acknowledgements

# Romain Tavenard
# License: BSD 3 clause

import numpy
import matplotlib.pyplot as plt

from tslearn.clustering import TimeSeriesKMeans
from tslearn.datasets import CachedDatasets
from tslearn.preprocessing import TimeSeriesScalerMeanVariance,TimeSeriesResampler

print("libraries imported")


dpath='/media/ashutosh/Data/Leonard/JavaIsland/Data'

ts_parameters=mat73.loadmat(dpath+'/java_losTS.mat')

ts=ts_parameters['java_TS']
# static_parameters=spio.loadmat('/media/ashutosh/Data/Project02_LosAngeles/Features/rs_static_parameters.mat')

j=1

frame=ts[j]

elpx_ll=frame[0]

lonlat=elpx_ll[:,0:2]

mvel=frame[1]
images=frame[2]

rts=frame[3]

rts=np.delete(rts, 0, axis=1)

inc_angle=elpx_ll[:,2:4] #containe lat lon and incidence angle

#displaying size of some of the parameters to see dimensional consistency

print(mvel)

# Find indices of NaN values
nan_indices = np.isnan(mvel)

# Remove rows with NaN values
mvel = mvel[~nan_indices]

rts=rts[~nan_indices,:]

lonlat=lonlat[~nan_indices,:]

elpx_ll=elpx_ll[~nan_indices,:]

inc_angle=inc_angle[~nan_indices,:]


X_static=np.column_stack((mvel,inc_angle)) #creating common input for the LSTM models


#print("Shape of phase time series", phase_sts.shape)

print("Shape of time series displacement", rts.shape)



X=rts


import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, lonlat, test_size=0.7, random_state=42)

#computing distance to avoid conflicts in space

sz = X_train.shape[1]

X_train=X_train.reshape((X_train.shape[0],X_train.shape[1],1))


#Reference for time series kmeans
#https://tslearn.readthedocs.io/en/stable/auto_examples/clustering/plot_kmeans.html#sphx-glr-auto-examples-clustering-plot-kmeans-py


#Finding optimum number of clusters    


# # Define the desired size of the downsampled vector
# downsampled_size = int(mvel.shape[0]/2)  # You can adjust this based on your needs

# # downsampled_mvel = downsampled_mvel[~np.isnan(downsampled_mvel)]

# downsampled_mvel=downsampled_mvel.reshape(-1,1)


from tslearn.clustering import TimeSeriesKMeans
# Preprocessing
from sklearn.preprocessing import MinMaxScaler
# Algorithms
# from minisom import MiniSom
from tslearn.barycenters import dtw_barycenter_averaging
from tslearn.clustering import TimeSeriesKMeans
from sklearn.cluster import KMeans
# Elbow Method for K means
# Import ElbowVisualizer
from yellowbrick.cluster import KElbowVisualizer
model = KMeans()
# k is range of number of clusters.
visualizer = KElbowVisualizer(model, k=(2,10), timings= True)
visualizer.fit(X_static)        # Fit data to visualizer
visualizer.show()        # Finalize and render figure

optimal_cluster_value = visualizer.elbow_value_

# Store the optimum cluster value
# optimal_cluster_value = visualizer.k_elbow_value_

# Now you can use optimal_cluster_value in your further analysis or clustering
print("Optimal number of clusters:", optimal_cluster_value)


# Silhouette Score for K means
# Import necessary libraries
from yellowbrick.cluster import KElbowVisualizer
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import numpy as np

# # Create a range of cluster values
# k_values = range(2, 10)

# # Store silhouette scores for each k value
# silhouette_scores = []

# # Iterate over each k value
# for k in k_values:
#     # Create KMeans model
#     kmeans = KMeans(n_clusters=k)
#     # Fit the model
#     kmeans.fit(downsampled_mvel)
#     # Get cluster labels
#     labels = kmeans.labels_
#     # Calculate silhouette score
#     silhouette_avg = silhouette_score(downsampled_mvel, labels)
#     # Append the silhouette score to the list
#     silhouette_scores.append(silhouette_avg)

# # Find the index of the maximum silhouette score
# optimal_cluster_index_silhouette = np.argmax(silhouette_scores)

# # Obtain the corresponding number of clusters
# optimal_cluster_value_silhouette = k_values[optimal_cluster_index_silhouette]

# # Now you can use optimal_cluster_value_silhouette in your further analysis or clustering
# print("Optimal number of clusters based on Silhouette score:", optimal_cluster_value_silhouette)

from tslearn.clustering import TimeSeriesKMeans

seed = 0
numpy.random.seed(seed)

'''
X_train, y_train, X_test, y_test = CachedDatasets().load_dataset("Trace")
X_train = X_train[y_train < 4]  # Keep first 3 classes

'''

# DBA-k-means
print("DBA k-means")
dba_km = TimeSeriesKMeans(n_clusters=optimal_cluster_value,
                          n_init=2,
                          n_jobs=10,
                          metric="dtw",
                          verbose=True,
                          max_iter_barycenter=5,
                          random_state=seed)
y_pred_dba = dba_km.fit_predict(X_train)

y_pred_dba_X=dba_km.predict(X)


spio.savemat(dpath+'/rs_predicted_ts_clusters_frame2.mat', dict(y_pred_dba_X=y_pred_dba_X, lonlat=lonlat))


# # Soft-DTW-k-means

# seed = 0
# numpy.random.seed(seed)
# print("Soft-DTW k-means")
# sdtw_km = TimeSeriesKMeans(n_clusters=optimal_cluster_value,
#                            metric="softdtw",
#                            n_jobs=10,
#                            metric_params={"gamma": .01},
#                            verbose=True,
#                            random_state=seed)
# y_pred_sdtw = sdtw_km.fit_predict(X_train)

# for yi in range(3):
#     plt.subplot(3, 3, 7 + yi)
#     for xx in X_train[y_pred_sdtw == yi]:
#         plt.plot(xx.ravel(), "k-", alpha=.2)
#     plt.plot(sdtw_km.cluster_centers_[yi].ravel(), "r-")
#     plt.xlim(0, sz)
#     plt.ylim(-4, 4)
#     plt.text(0.55, 0.85,'Cluster %d' % (yi + 1),
#              transform=plt.gca().transAxes)
#     if yi == 1:
#         plt.title("Soft-DTW $k$-means")

# plt.tight_layout()
# plt.show()

# mdic_softdtw = {"y_pred_sdtw": y_pred_sdtw, "X_train": X_train}

# spio.savemat('predicted_softdtw.mat', mdic_softdtw)


# # Visualization for obtained clusters   
# def plot_clusters(data, cluster_model, dim_red_algo):
#     """
#     Plots clusters obtained by clustering model 

#     data: pd.DataFrame or np.array
#         Time Series Data
#     cluster_model: Class
#         Clustering algorithm 
#     dim_red_algo: Class
#         Dimensionality reduction algorithm (e.g. TSNE/PCA/MDS...) 
#     Returns:
#     -------
#     None
#     """
#     cluster_labels = cluster_model.fit_predict(data)
#     centroids = cluster_model.cluster_centers_
#     u_labels = np.unique(cluster_labels)
    
#     # Centroids Visualization
#     plt.figure(figsize=(16, 10))
#     plt.scatter(centroids[:, 0] , centroids[:, 1] , s=150, color='r', marker="x")
    
#     # Downsize the data into 2D
#     if data.shape[1] > 2:
#         data_2d = dim_red_algo.fit_transform(data)
#         for u_label in u_labels:
#             cluster_points = data[(cluster_labels == u_label)]
#             plt.scatter(cluster_points[:, 0], cluster_points[:, 1], label=u_label)
#     else:
#         for u_label in u_labels:
#             cluster_points = data[(cluster_labels == u_label)]
#             plt.scatter(cluster_points[:, 0], cluster_points[:, 1], label=u_label)

#     plt.title('Clustered Data')
#     plt.xlabel("Feature space for the 1st feature")
#     plt.ylabel("Feature space for the 2nd feature")
#     plt.grid(True)
#     plt.legend(title='Cluster Labels');

