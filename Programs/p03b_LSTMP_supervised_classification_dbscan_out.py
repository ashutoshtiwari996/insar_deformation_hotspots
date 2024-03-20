#!/usr/bin/env python
# coding: utf-8


"""
A program to perform supervised clustering using LSTM and perceptron learning networks
given dbscan clustering labels as training labels

"""

from matplotlib import pyplot
import numpy as np
import scipy.io as spio
import mat73
import sklearn
import pandas
# from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import cross_val_score, KFold
# from keras.utils.vis_utils import plot_model
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from keras.models import Model
from keras.layers import Input
from keras.layers import LSTM,Bidirectional,Concatenate
from keras.layers import Dense, Dropout
# from keras.utils.vis_utils import plot_model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras import layers
from tensorflow.keras import regularizers


#get one-hot encoding
#reading dataset

from os import listdir

from keras.preprocessing import sequence
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.optimizers import Adam
from keras.models import load_model
from keras.callbacks import ModelCheckpoint

# Preprocessing
from sklearn.preprocessing import MinMaxScaler
# Algorithms
from minisom import MiniSom
from tslearn.barycenters import dtw_barycenter_averaging
from tslearn.clustering import TimeSeriesKMeans
from sklearn.cluster import KMeans

import scipy.io as spio
import sktime


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

"""
RNN + SLP Model

"""

# Define input layer
recurrent_input = Input(shape=(n_timesteps,n_features),name="TIMESERIES_INPUT")
static_input = Input(shape=(X_static.shape[1], ),name="STATIC_INPUT")


#RNN Layers
# layer - 1

rec_layer_one = Bidirectional(LSTM(128, kernel_regularizer=regularizers.l1_l2(l1=0, l2=0.01), recurrent_regularizer=regularizers.l1_l2(l1=0, l2=0.01),return_sequences=True),name ="BIDIRECTIONAL_LAYER_1")(recurrent_input)
rec_layer_one = Dropout(0.2,name ="DROPOUT_LAYER_1")(rec_layer_one)# layer - 2
rec_layer_two = Bidirectional(LSTM(64, kernel_regularizer=regularizers.l1_l2(l1=0, l2=0.01), recurrent_regularizer=regularizers.l1_l2(l1=0, l2=0.01)),name ="BIDIRECTIONAL_LAYER_2")(rec_layer_one)
rec_layer_two = Dropout(0.2,name ="DROPOUT_LAYER_2")(rec_layer_two)

# SLP Layers
static_layer_one = Dense(128,  kernel_regularizer=regularizers.l1_l2(l1=0, l2=0.01),  activation='selu',name="DENSE_LAYER_1")(static_input)

# Combine layers - RNN + SLP
combined = Concatenate(axis= 1,name = "CONCATENATED_TIMESERIES_STATIC")([rec_layer_two,static_layer_one])
combined_dense_two = Dense(128, activation='relu',name="DENSE_LAYER_2")(combined)
output = Dense(n_outputs,activation='sigmoid',name="OUTPUT_LAYER")(combined_dense_two)

# Compile ModeL
model = Model(inputs=[recurrent_input,static_input],outputs=[output])


from keras import backend as K

# Define metrics for evaluating the model - recall, precision and f1-score

# Define metrics for evaluating the model - recall, precision and f1-score
def recall_m(y_true, y_pred):
  true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
  possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
  recall = true_positives / (possible_positives + K.epsilon())
  return recall

def precision_m(y_true, y_pred):
  true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
  predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
  precision = true_positives / (predicted_positives + K.epsilon())
  return precision

def f1_m(y_true, y_pred):
  precision = precision_m(y_true, y_pred)
  recall = recall_m(y_true, y_pred)
  return 2*((precision*recall)/(precision+recall+K.epsilon()))

adam = Adam(learning_rate=0.001)                          
# binary cross entropy loss
# model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy',f1_m,precision_m, recall_m])

# binary cross entropy loss
model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy',f1_m])

# model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy',f1_m])

# fit network

history =  model.fit([np.asarray(X_train).astype('float32'), np.asarray(X_static).astype('float32')],y_train_onehot, epochs=5, batch_size=64, verbose=1)# summarize history for accuracypyplot.plot(history.history['accuracy'])

y_pred_lstmp=model.predict([np.asarray(X_train).astype('float32'), np.asarray(X_static).astype('float32')])

y_pred_lstmp_clusters=model.predict([X_train, X_static])

y_pred_lstmp_class=y_pred_lstmp_clusters.argmax(axis=-1)

spio.savemat(ppath+'/rs_lstmp_clustering.mat', dict(y_pred_lstmp_class=y_pred_lstmp_class))

#Save the model

from keras.models import model_from_json

# serialize classifier to JSON
classifier_json = model.to_json()
with open("rs_lstmp_clustering_model.json", "w") as json_file:
    json_file.write(classifier_json)
# serialize weights to HDF5
classifier_json.save_weights("rs_lstmp_clustering_model.h5")
print("Saved classifier to disk")

model.save_weights("rs_lstmp_clustering_model.tf",save_format='tf')

pyplot.plot(history.history['val_accuracy'])
pyplot.title('model accuracy')
pyplot.ylabel('accuracy')
pyplot.xlabel('epoch')
pyplot.legend(['train', 'validation'], loc='upper left')
pyplot.show()# summarize history for losspyplot.plot(history.history['loss'])
pyplot.plot(history.history['val_loss'])
pyplot.title('model loss')
pyplot.ylabel('loss')
pyplot.xlabel('epoch')
pyplot.legend(['train', 'validation'], loc='upper left')
pyplot.show()#evaluate modelloss, accuracy, f1_score, precision, recall = model.evaluate([np.asarray(x_test_reshape).astype('float32'),np.asarray(x_test_static).astype('float32')], y_test_reshape, batch_size=batch_size, verbose=0)#print outputprint("Accuracy:{} , F1_Score:{}, Precision:{}, Recall:{}".format(accuracy, f1_score, precision, recall))

