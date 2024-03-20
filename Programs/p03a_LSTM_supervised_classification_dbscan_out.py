#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 10 15:16:04 2022

@author: ashutosh
"""

"""
A program to perform supervised clustering using LSTM and perceptron learning networks
given dbscan clustering labels as training labels

"""

from matplotlib import pyplot
import numpy as np
import scipy.io as spio
import mat73
import sklearn
import pandas as pd
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import cross_val_score, KFold
from keras.utils.vis_utils import plot_model
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from keras.models import Model
from keras.layers import Input
from keras.layers import LSTM,Bidirectional,Concatenate
from keras.layers import Dense, Dropout
from keras.utils.vis_utils import plot_model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras import layers
from tensorflow.keras import regularizers
from keras.preprocessing import sequence
import tensorflow as tf
from keras.optimizers import Adam
from keras.models import load_model
from keras.callbacks import ModelCheckpoint
from os import listdir
import scipy.io as spio
import sktime

#get one-hot encoding
#reading dataset

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

##reading datasets

ppath='/media/projects/insar_deformation_hotspots/Data'
#reading dataset
#reading dataset

ts_parameters=mat73.loadmat(ppath+'/R_TS_den.mat')

mean_velocity=spio.loadmat(ppath+'/MVel.mat')

mean_velocity_std=spio.loadmat(ppath+'/MVelq.mat')

incidence_angle=spio.loadmat(ppath+'/inc_angle.mat')

coherence=spio.loadmat(ppath+'/Pco_m.mat')

lonlat=spio.loadmat(ppath+'/elpx_ll.mat')
  
elpx_ll=lonlat['elpx_ll']

lonlat=np.delete(elpx_ll, 2, axis=1)

dbscan_out=spio.loadmat(ppath+'/output_dbscan.mat')
# read matrices from data dictionaries

rts=ts_parameters['R_TS_den'].astype('float16') #time series of n time steps

mvel=mean_velocity['MVel'].astype('float16') #mean LOS velocity

mvelq=mean_velocity_std['MVelq'].astype('float16') #mean LOS velocity

inc_angle=incidence_angle['inc_angle'].astype('float16') #containe lat lon and incidence angle

coh=coherence['Pco_m']

#displaying size of some of the parameters to see dimensional consistency

print(mvel)


X_static=np.column_stack((mvel, mvelq, inc_angle, coh)) #creating common input for the LSTM models
y=dbscan_out['y_predn_full']

rts=ts_parameters['R_TS_den'] #time series of n time steps

#rtsq=ts_parameters['R_TSq']  #may contain nans

print("Shape of time series displacement", rts.shape)

X=rts


X_train=X.reshape((X.shape[0],X.shape[1],1))
y_train=y

from keras.utils.np_utils import to_categorical


n_clusters=np.unique(y)

y_train_onehot=to_categorical(y,num_classes=n_clusters)

# y_train_onehot=to_categorical(y)
#y_test_new=to_categorical(y_test)

sz = X_train.shape[1]


import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


# Define timesteps and the number of features

n_timesteps = X_train.shape[1]

n_features = X_train.shape[2]

n_outputs=np.unique(y_train)

#model building

model = Sequential()
model.add(LSTM(256, input_shape=(sz, 1)))
model.add(Dense(n_outputs, activation='softmax'))


adam = Adam(learning_rate=0.001)
chk = ModelCheckpoint('best_model_lstm_clusters.pkl', monitor='val_acc', save_best_only=True, mode='max', verbose=1)
model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])
history=model.fit(X_train, y_train_onehot, epochs=5, batch_size=64, callbacks=[chk])

# plt.plot(history.history['accuracy'])
plt.plot(history.history['loss'])
plt.title('Model Training', fontweight='bold', fontsize=14)
plt.ylabel('Loss', fontweight='bold', fontsize=14)
plt.xlabel('Epoch', fontweight='bold', fontsize=14)
plt.legend(['Training Loss'], fontsize=14, loc='upper right')

# plt.legend(['training', 'validation'], loc='upper right')
plt.show()


y_pred_lstm=model.predict(X_train)

y_pred_lstm_class=y_pred_lstm.argmax(axis=-1)

spio.savemat('rs_lstm_clustering.mat', dict(y_pred_lstm_class=y_pred_lstm_class))


#Save the model

from keras.models import model_from_json

# serialize classifier to JSON
classifier_json = model.to_json()
with open("rs_lstm_clustering_model.json", "w") as json_file:
    json_file.write(classifier_json)
# serialize weights to HDF5

model.save_weights(ppath+"/rs_lstm_clustering_model.tf",save_format='tf')

# classifier_json.save_weights("rs_lstm_classification.h5")
# print("Saved classifier to disk")

#Now train from loaded model

# load json and create classifier
json_file = open('rs_lstm_clustering_model.json', 'r')
loaded_classifier_json = json_file.read()
json_file.close()
loaded_classifier = model_from_json(loaded_classifier_json)
# load weights into new classifier
loaded_classifier.load_weights("rs_lstm_clustering_model.tf")
print("Loaded classifier from disk")

#loading the model and checking accuracy on the test data
model = load_model('rs_lstm_clustering_model.pkl')

# from sklearn.metrics import accuracy_score
# test_preds = model.predict_classes(X)
# accuracy_score(y, test_preds)

