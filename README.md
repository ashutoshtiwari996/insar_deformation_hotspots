# insar_deformation_hotspots
Deriving spatio-temporal deformation patterns over LA using InSAR and AI​

The set of programs can be used for post-processing InSAR deformation maps obtained from multi-temporal InSAR processing.

Input: Time series deformation matrix in .mat format (exported from WabInSAR software developed by Manoochehr Shirzaei). WabInSAR software is open access and can be downloaded from the following link: https://sites.google.com/vt.edu/eadar-lab/software

Output: Cluster maps based on time series k-means, dbscan, followed by codes to remove unwanted clusters and merging clusters. 

Instructions for running the program:
Use requirements.txt to install necessary libraries for the program. Follow these steps:

In your shell (or command prompt)

(i) Go to the directory where requirements.txt is located (ii) activate your virtualenv (if you create a separate virtual envcironment for this program) (iii) run pip install -r requirements.txt

The input data for this program can be InSAR displacement time series. The output labels for deep learning methods can be the cluster labels generated from the output of DBSCAN algorithm. 

Please cite the following if using the program and data:
Tiwari, Ashutosh; Shirzaei, Manoochehr (2024). A novel machine learning and deep learning semi-supervised approach for automatic detection of InSAR-based deformation hotspots, International Journal of Applied Earth Observation and Geoinformation Volume 126, DOI: [10.1016/j.jag.2023.103611]
