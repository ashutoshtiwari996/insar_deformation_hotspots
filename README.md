# insar_deformation_hotspots

# Deriving spatio-temporal deformation patterns over LA using InSAR and AI​

The set of programs can be used for post-processing InSAR deformation maps obtained from multi-temporal InSAR processing.

**Input**: Time series deformation matrix in .mat format (exported from WabInSAR software developed by Manoochehr Shirzaei). WabInSAR software is open access and can be downloaded from the following link: https://sites.google.com/vt.edu/eadar-lab/software

**Output**: Cluster maps based on time series k-means, dbscan, followed by codes to remove unwanted clusters and merging clusters. 

Instructions for running the program:
Use requirements.txt to install necessary libraries for the program. Follow these steps:

In your shell (or command prompt)

(i) Go to the directory where requirements.txt is located (ii) activate your virtualenv (if you create a separate virtual envcironment for this program) (iii) run pip install -r requirements.txt

The input data for this program can be InSAR displacement time series. The output labels for deep learning methods can be the cluster labels generated from the output of DBSCAN algorithm. 


**Set of programs for information mining on InSAR derived deformation maps**

For python programs, please install necassary libraries using 
Use requirements.txt to install necessary libraries for the python programs.

Follow these steps:

In your shell (or command prompt)

(i) Go to the directory where requirements.txt is located (ii) activate your virtualenv (if you create a separate virtual envcironment for this program) (iii) run pip install -r requirements.txt

## Sequence of programs and usage

p01_time_series_unsupervised_clustering_rts.py: This program perform time series clustering for InSAR time series displacement using time series k-means algorithm

p02_Individual_cluster_dbscan_clustering.py: This program performs spatial clustering over individual temporal clusters detected using p01.

p03_plot_dbscan_clusters.m: A program for plotting the resulting clusters from p02. 

p03a_LSTM_supervised_classification_dbscan_out.py: Program for supervised classification using LSTM networks with clusters generated from results of p02 acting as training labels. 

p03b_LSTMP_supervised_classification_dbscan_out.py: Program for supervised classification using LSTM and perceptron networks with clusters generated from results of p02 acting as training labels.

p04_finding_spatial_correlation_lengths.m: Finding spatial correlation lengths for displacement in the study area, to find out a threshold for removing widespread unwanted clusters.

p05_spatial_dispersion.m: Finding spatial dispersion of individual clusters generated from p02. Those with large dispersions would be removed.

## About the data files

elpx_ll.mat: Coordinates for elite pixels

MVel.mat: One dimensional line of sight velocities

MVelq.mat: Standard deviation for line of sight velocities

Pco_m.mat: Mean coherence

R_TS_den.mat: Displacement time series

## Citing our work

Please cite the following if using the program and data:
Tiwari, Ashutosh; Shirzaei, Manoochehr (2024). A novel machine learning and deep learning semi-supervised approach for automatic detection of InSAR-based deformation hotspots, International Journal of Applied Earth Observation and Geoinformation Volume 126, DOI: [10.1016/j.jag.2023.103611]
