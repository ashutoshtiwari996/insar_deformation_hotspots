#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan  2 18:39:23 2023

@author: ashutosh
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov  4 14:27:32 2022

@author: ashutosh
"""

# Native libraries
import os
import math

# Essential Libraries
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import numpy as np
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt
import scipy.io as spio
import mat73


# Preprocessing
from sklearn.preprocessing import MinMaxScaler

# Algorithms
from minisom import MiniSom
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA


#Acknowledgements

# Romain Tavenard
# License: BSD 3 clause

print("libraries imported")

#reading InSAR dataset


"""
Set the directory path where you download your data

"""

ppath='/media/projects/insar_deformation_hotspots/Data'

# ppath='/media/projects/insar_deformation_hotspots-main/Data'

#reading dataset

# ts_parameters=mat73.loadmat(dpath+'/R_TS_den.mat')

tskmeans=spio.loadmat(ppath+'/rs_predicted_clusters_dba.mat')

ts_parameters=mat73.loadmat(ppath+'/R_TS_den.mat')

rts=ts_parameters['R_TS_den']
# static_parameters=spio.loadmat('/media/ashutosh/Data/Project02_LosAngeles/Features/rs_static_parameters.mat')


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import matplotlib.colors as colors

from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture

# affinity propagation clustering
from numpy import unique
from numpy import where
from sklearn.datasets import make_classification
from sklearn.cluster import AffinityPropagation
from sklearn.cluster import AgglomerativeClustering
from matplotlib import pyplot


y=tskmeans['y_pred_dba_X']

y=np.transpose(y)

mean_velocity=spio.loadmat(ppath+'/MVel.mat')

mvel=mean_velocity['MVel'].astype('float16') #mean LOS velocity

lonlat=spio.loadmat(ppath+'/elpx_ll.mat')
  
elpx_ll=lonlat['elpx_ll']

lonlat=np.delete(elpx_ll, 2, axis=1)
    
########################################################################

# y=np.transpose(yhat_gmm); #using results from gmm

y = y.astype('float64') #converting integer cluster ids to float values for ease of operations

#lonlat=elpx_ll(:,1:2);

#trying to find individual cluster ids

id1=np.where(y==0)
id2=np.where(y==1)
id3=np.where(y==2)
id4=np.where(y==3)
# id5=np.where(y==4)
# id6=np.where(y==5)
# id7=np.where(y==6)

id1=np.array(id1)
id1=np.transpose(id1)
id1=id1[:,0]

id2=np.array(id2)
id2=np.transpose(id2)
id2=id2[:,0]

id3=np.array(id3)
id3=np.transpose(id3)
id3=id3[:,0]

id4=np.array(id4)
id4=np.transpose(id4)
id4=id4[:,0]

# id5=np.array(id5)
# id5=np.transpose(id5)
# id5=id5[:,0]

# id6=np.array(id6)
# id6=np.transpose(id6)
# id6=id6[:,0]

# id7=np.array(id7)
# id7=np.transpose(id7)
# id7=id7[:,0]


y1=y[id1]
y2=y[id2]
y3=y[id3]
y4=y[id4]
# y5=y[id5]
# y6=y[id6]
# y7=y[id7]


lonlat1=lonlat[id1,:];
lonlat2=lonlat[id2,:];
lonlat3=lonlat[id3,:];
lonlat4=lonlat[id4,:];
# lonlat5=lonlat[id5,:];
# lonlat6=lonlat[id6,:];
# lonlat7=lonlat[id7,:];


rts1=rts[id1,:]
rts2=rts[id2,:]
rts3=rts[id3,:]
rts4=rts[id4,:]
# rts5=rts[id5,:]
# rts6=rts[id6,:]
# rts7=rts[id7,:]

# #%Training input for supervised learning

# rts_full=np.row_stack((rts1,rts2,rts3,rts4,rts5,rts6,rts7))


# pco1=Pco_m[id1]
# pco2=Pco_m[id2]
# pco3=Pco_m[id3]
# pco4=Pco_m[id4]
# pco5=Pco_m[id5]
# pco6=Pco_m[id6]
# pco7=Pco_m[id7]


# pco_full=np.row_stack((pco1,pco2,pco3,pco4,pco5,pco6,pco7))

mvel1=mvel[id1]
mvel2=mvel[id2]
mvel3=mvel[id3]
mvel4=mvel[id4]
# mvel5=MVel[id5]
# mvel6=MVel[id6]
# mvel7=MVel[id7]

# MVel_full=np.row_stack((mvel1,mvel2,mvel3,mvel4,mvel5,mvel6,mvel7))

# mvelq1=MVelq[id1]
# mvelq2=MVelq[id2]
# mvelq3=MVelq[id3]
# mvelq4=MVelq[id4]
# mvelq5=MVelq[id5]
# mvelq6=MVelq[id6]
# mvelq7=MVelq[id7]

# MVelq_full=np.row_stack((mvelq1,mvelq2,mvelq3,mvelq4,mvelq5,mvelq6,mvelq7))


# inc1=inc_angle[id1]
# inc2=inc_angle[id2]
# inc3=inc_angle[id3]
# inc4=inc_angle[id4]
# inc5=inc_angle[id5]
# inc6=inc_angle[id6]
# inc7=inc_angle[id7]

# inc_full=np.row_stack((inc1,inc2,inc3,inc4,inc5,inc6,inc7))

#spio.savemat('/media/ashutosh/Data/Project02_LosAngeles/output/static_input_dbscan_pyth.mat', 'lonlat_full', 'MVel_full','MVelq_full', 'inc_full', 'y_full', 'pco_full');


# save('../Features/ts_input_dbscan.mat', 'rts_full', '-v7.3');


rs1=np.column_stack((lonlat1,y1));
rs2=np.column_stack((lonlat2,y2));
rs3=np.column_stack((lonlat3,y3));
rs4=np.column_stack((lonlat4,y4));
# rs5=np.column_stack((lonlat5,y5));
# rs6=np.column_stack((lonlat6,y6));
# rs7=np.column_stack((lonlat7,y7));

rs_full=np.row_stack((rs1,rs2,rs3,rs4)) #,rs5,rs6,rs7))

# rs_full=np.row_stack((rs1,rs2,rs3,rs4, rs5)) #,rs5,rs6,rs7))


#This will be applied to the output of individual clusters 

# %trying the dbscan clustering over coordinates of individual clusters

#Ref for DB Scan algorithm: https://www.section.io/engineering-education/dbscan-clustering-in-python/

from sklearn.cluster import DBSCAN

#cluster 1


plt.scatter(lonlat1[:, 0], lonlat1[:,1], c = y1, cmap= "plasma") # plotting the clusters
plt.xlabel("Longitude") # X-axis label
plt.ylabel("Latitude") # Y-axis label
plt.show() 

epsilon1=0.05;

minpts1=500;

dbscan1 = DBSCAN(epsilon1, min_samples=minpts1).fit(lonlat1) 

idx1 = dbscan1.labels_

plt.scatter(lonlat1[:, 0], lonlat1[:,1], c = idx1, cmap= "plasma") # plotting the clusters
plt.xlabel("Longitude") # X-axis label
plt.ylabel("Latitude") # Y-axis label
plt.show() 


#cluster 2

plt.scatter(lonlat2[:, 0], lonlat2[:,1], c = y2, cmap= "plasma") # plotting the clusters
plt.xlabel("Longitude") # X-axis label
plt.ylabel("Latitude") # Y-axis label
plt.show() 


epsilon2=0.05;

minpts2=500;

dbscan2 = DBSCAN(epsilon2, min_samples=minpts2).fit(lonlat2) 

idx2 = dbscan2.labels_


plt.scatter(lonlat2[:, 0], lonlat2[:,1], c = idx2, cmap= "plasma") # plotting the clusters
plt.xlabel("Longitude") # X-axis label
plt.ylabel("Latitude") # Y-axis label
plt.show() 


#cluster 3

plt.scatter(lonlat3[:, 0], lonlat3[:,1], c = y3, cmap= "plasma") # plotting the clusters
plt.xlabel("Longitude") # X-axis label
plt.ylabel("Latitude") # Y-axis label
plt.show() 


epsilon3=0.05;

minpts3=500;

dbscan3 = DBSCAN(epsilon3, min_samples=minpts3).fit(lonlat3) 

idx3 = dbscan3.labels_


plt.scatter(lonlat3[:, 0], lonlat3[:,1], c = idx3, cmap= "plasma") # plotting the clusters
plt.xlabel("Longitude") # X-axis label
plt.ylabel("Latitude") # Y-axis label
plt.show() 


#cluster 4

plt.scatter(lonlat4[:, 0], lonlat4[:,1], c = y4, cmap= "plasma") # plotting the clusters
plt.xlabel("Longitude") # X-axis label
plt.ylabel("Latitude") # Y-axis label
plt.show() 


epsilon4=0.05;

minpts4=500;

dbscan4 = DBSCAN(epsilon4, min_samples=minpts4).fit(lonlat4) 

idx4 = dbscan4.labels_

plt.scatter(lonlat4[:, 0], lonlat4[:,1], c = idx4, cmap= "plasma") # plotting the clusters
plt.xlabel("Longitude") # X-axis label
plt.ylabel("Latitude") # Y-axis label
plt.show() 



# #cluster 5

# plt.scatter(lonlat5[:, 0], lonlat5[:,1], c = y5, cmap= "plasma") # plotting the clusters
# plt.xlabel("Longitude") # X-axis label
# plt.ylabel("Latitude") # Y-axis label
# plt.show() 


# epsilon5=0.02;

# minpts5=500;

# dbscan5 = DBSCAN(epsilon5, min_samples=minpts5).fit(lonlat5) 

# idx5 = dbscan5.labels_

# plt.scatter(lonlat5[:, 0], lonlat5[:,1], c = idx5, cmap= "plasma") # plotting the clusters
# plt.xlabel("Longitude") # X-axis label
# plt.ylabel("Latitude") # Y-axis label
# plt.show() 


# Filter out noise points (points with label -1) from each cluster
filtered_lonlat1 = lonlat1[idx1 != -1]
filtered_idx1 = idx1[idx1 != -1]

filtered_lonlat2 = lonlat2[idx2 != -1]
filtered_idx2 = idx2[idx2 != -1]

filtered_lonlat3 = lonlat3[idx3 != -1]
filtered_idx3 = idx3[idx3 != -1]

filtered_lonlat4 = lonlat4[idx4 != -1]
filtered_idx4 = idx4[idx4 != -1]

# filtered_lonlat5 = lonlat5[idx5 != -1]
# filtered_idx5 = idx5[idx5 != -1]


# Remove rows based on filtered idx for mvel1
mvel1_filtered = mvel1[idx1 != -1]

# Remove rows based on filtered idx for mvel2
mvel2_filtered = mvel2[idx2 != -1]

# Remove rows based on filtered idx for mvel3
mvel3_filtered = mvel3[idx3 != -1]

# Remove rows based on filtered idx for mvel4
mvel4_filtered = mvel4[idx4 != -1]


# Remove rows based on filtered idx for rts1
rts1_filtered = rts1[idx1 != -1]

# Remove rows based on filtered idx for rts2
rts2_filtered = rts2[idx2 != -1]

# Remove rows based on filtered idx for rts3
rts3_filtered = rts3[idx3 != -1]

# Remove rows based on filtered idx for rts4
rts4_filtered = rts4[idx4 != -1]

# rts5_filtered = rts5[idx5 != -1]

# Plot the filtered clusters
plt.scatter(filtered_lonlat1[:, 0], filtered_lonlat1[:,1], c = filtered_idx1, cmap= "plasma") 
plt.xlabel("Longitude") 
plt.ylabel("Latitude") 
plt.show() 

plt.scatter(filtered_lonlat2[:, 0], filtered_lonlat2[:,1], c = filtered_idx2, cmap= "plasma") 
plt.xlabel("Longitude") 
plt.ylabel("Latitude") 
plt.show() 

plt.scatter(filtered_lonlat3[:, 0], filtered_lonlat3[:,1], c = filtered_idx3, cmap= "plasma") 
plt.xlabel("Longitude") 
plt.ylabel("Latitude") 
plt.show() 

plt.scatter(filtered_lonlat4[:, 0], filtered_lonlat4[:,1], c = filtered_idx4, cmap= "plasma") 
plt.xlabel("Longitude") 
plt.ylabel("Latitude") 
plt.show() 

# plt.scatter(filtered_lonlat5[:, 0], filtered_lonlat5[:,1], c = filtered_idx5, cmap= "plasma") 
# plt.xlabel("Longitude") 
# plt.ylabel("Latitude") 
# plt.show() 


# Combine labels and lonlat matrices for each cluster
idx1n = filtered_idx1
idx2n = filtered_idx2 + np.max(idx1n) + 1
idx3n = filtered_idx3 + np.max(idx2n) + 1
idx4n = filtered_idx4 + np.max(idx3n) + 1
# idx5n = filtered_idx5 + np.max(idx4n) + 1


mvel_full=np.vstack((mvel1_filtered, mvel2_filtered, mvel3_filtered, mvel4_filtered))


rts_full=np.vstack((rts1_filtered, rts2_filtered, rts3_filtered, rts4_filtered))


# Concatenate all the labels into a single array
y_predn_full = np.hstack((idx1n, idx2n, idx3n, idx4n))#, idx5n))

# Concatenate all the lonlat matrices into a single array
lonlat_full = np.vstack((filtered_lonlat1, filtered_lonlat2, filtered_lonlat3, filtered_lonlat4)) #, filtered_lonlat5))


# # Concatenate all the labels into a single array
# y_predn_full = np.hstack((idx1n, idx2n, idx3n, idx4n, idx5n))

# # Concatenate all the lonlat matrices into a single array
# lonlat_full = np.vstack((filtered_lonlat1, filtered_lonlat2, filtered_lonlat3, filtered_lonlat4, filtered_lonlat5))



# Combine lonlat matrices with corresponding labels
rs_db_full = np.column_stack((lonlat_full, rts_full, y_predn_full))


#display

plt.scatter(lonlat_full[:, 0], lonlat_full[:,1], c = y_predn_full, cmap= "plasma") # plotting the clusters
plt.xlabel("Longitude") # X-axis label
plt.ylabel("Latitude") # Y-axis label
plt.show() 


spio.savemat(ppath+'/output_dbscan.mat', {"mvel_full":mvel_full, "rs_db_full":rs_db_full, "lonlat_full": lonlat_full, "y_predn_full": y_predn_full, "rs_full": rs_full, "lonlat":lonlat});


