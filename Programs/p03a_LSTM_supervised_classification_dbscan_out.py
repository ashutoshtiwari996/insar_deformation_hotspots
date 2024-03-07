#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 10 15:16:04 2022

@author: ashutosh
"""

import pandas as pd
import numpy as np
#matplotlib inline
import matplotlib.pyplot as plt
from os import listdir

from keras.preprocessing import sequence
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM

from keras.optimizers import Adam
from keras.models import load_model
from keras.callbacks import ModelCheckpoint


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


import numpy
import matplotlib.pyplot as plt


os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

##reading datasets


#reading dataset

#parameters=spio.loadmat('/media/ashutosh/Data/Project02_LosAngeles/S1_descending/input_parameters.mat')

#for -v7.3 files

ts_parameters=mat73.loadmat('/media/ashutosh/Data/Project02_LosAngeles/Features/ts_input_dbscan.mat')


static_parameters=spio.loadmat('/media/ashutosh/Data/Project02_LosAngeles/Features/static_input_dbscan.mat')

lables=spio.loadmat('/media/ashutosh/Data/Project02_LosAngeles/output/output_dbscan_single_level_33clusters_InSAR.mat')

lonlat=lables['lonlat_full']

y=lables['y_predn_full']

y=np.transpose(y)

print(ts_parameters)

#y=lables['y_pred_dba_X']

#y=np.transpose(y)

print(static_parameters)


#phase_sts=ts_parameters['Phase_sts_TS'] #interferometric phase time series

rts=ts_parameters['rts_full'] #time series of n time steps

#rtsq=ts_parameters['R_TSq']  #may contain nans

mean_vel=static_parameters['MVel_full'] #mean LOS velocity

mean_velq=static_parameters['MVelq_full'] #mean LOS velocity

inc_angle=static_parameters['inc_full'] #containe lat lon and incidence angle

#displaying size of some of the parameters to see dimensional consistency

print(mean_vel)

#dimensions of input parameters

print("Shape of mean_vel", mean_vel.shape)

#mean_vel= mean_vel[~np.isnan(mean_vel)]

#print("Shape of mean_vel after NaN removal", mean_vel.shape)

print("Shape of incidence angle", inc_angle.shape)

#inc_angle= inc_angle[~np.isnan(inc_angle)]

#print("Shape of inc_angle after NaN removal", inc_angle.shape)

X_static=np.column_stack((mean_vel, mean_velq, inc_angle)) #creating common input for the LSTM models


#print("Shape of phase time series", phase_sts.shape)

print("Shape of time series displacement", rts.shape)

X=rts


X_train=X

sz = X_train.shape[1]

X_train=X.reshape((X.shape[0],X.shape[1],1))


X=X.reshape((X.shape[0],X.shape[1],1))

X_train=X


from keras.utils.np_utils import to_categorical

n_clusters=33

y_train_onehot=to_categorical(y,num_classes=n_clusters)

import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

#model building

import pandas as pd
import numpy as np
#%matplotlib inline
import matplotlib.pyplot as plt
from os import listdir

from keras.preprocessing import sequence
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM

from keras.optimizers import Adam
from keras.models import load_model
from keras.callbacks import ModelCheckpoint

model = Sequential()
model.add(LSTM(256, input_shape=(sz, 1)))
model.add(Dense(33, activation='softmax'))


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


y_pred_lstm_classification=model.predict(X_train)

y_pred_lstm_class=y_pred_lstm_classification.argmax(axis=-1)

spio.savemat('rs_lstm_classification.mat', dict(y_pred_lstm_classification=y_pred_lstm_classification))

spio.savemat('rs_lstm_classification_class.mat', dict(y_pred_lstm_class=y_pred_lstm_class))


#Save the model

from keras.models import model_from_json

# serialize classifier to JSON
classifier_json = model.to_json()
with open("rs_lstm_classification.json", "w") as json_file:
    json_file.write(classifier_json)
# serialize weights to HDF5

model.save_weights("rs_lstm_classification.tf",save_format='tf')

# classifier_json.save_weights("rs_lstm_classification.h5")
# print("Saved classifier to disk")

#Now train from loaded model

# load json and create classifier
json_file = open('rs_lstm_classification.json', 'r')
loaded_classifier_json = json_file.read()
json_file.close()
loaded_classifier = model_from_json(loaded_classifier_json)
# load weights into new classifier
loaded_classifier.load_weights("rs_lstm_classification.tf")
print("Loaded classifier from disk")

#loading the model and checking accuracy on the test data
model = load_model('rs_lstm_classification.pkl')

from sklearn.metrics import accuracy_score
test_preds = model.predict_classes(X)
accuracy_score(y, test_preds)

