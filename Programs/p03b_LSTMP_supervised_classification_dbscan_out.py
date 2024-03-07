#!/usr/bin/env python
# coding: utf-8


from matplotlib import pyplot
import numpy as np
import scipy.io as spio
import mat73
import sklearn
import pandas
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import cross_val_score, KFold
from keras.utils.vis_utils import plot_model
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
#from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA
from keras.models import Model
from keras.layers import Input
from keras.layers import LSTM,Bidirectional,Concatenate
from keras.layers import Dense, Dropout
from keras.utils.vis_utils import plot_model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras import layers
from tensorflow.keras import regularizers


#reading dataset

#parameters=spio.loadmat('/media/ashutosh/Data/Project02_LosAngeles/S1_descending/input_parameters.mat')

#for -v7.3 files

parameters=mat73.loadmat('/media/ashutosh/Data/Project02_LosAngeles/S1_descending/input_parameters.mat')


aps=parameters['APS_TS'] #atmospheric phase time series

phase_sts=parameters['Phase_sts_TS0'] #interferometric phase time series

rts=parameters['R_TS_den'] #time series of n time steps

mean_vel=parameters['MVel'] #mean LOS velocity

elpx_ll=parameters['elpx_ll'] #containe lat lon and incidence angle

#displaying size of some of the parameters to see dimensional consistency

print(mean_vel)

print("Shape of mean_vel", mean_vel.shape)

#mean_vel= mean_vel[~np.isnan(mean_vel)]

#print("Shape of mean_vel after NaN removal", mean_vel.shape)

inc=elpx_ll[:,2] #incidence angle

#inc_angle=np.delete(inc_angle,[0,1],1)

print(inc)

print("Shape of inc_angle", inc.shape)

#inc_angle= inc_angle[~np.isnan(inc_angle)]

#print("Shape of inc_angle after NaN removal", inc_angle.shape)

X_static=np.column_stack((mean_vel, inc)) #creating common input for the LSTM models


# if data has one column with zero entries (master image), remove it


#phase_sts=parameters['Phase_sts_TS0']

#rts=parameters['R_TS0']

#phase_sts=np.delete(phase_sts,0,1) # removing master column which is zero from training

#rts=np.delete(rts,0,1) # removing master column which is zero from target



# Define timesteps and the number of features

n_timesteps = 246

n_features = 2

n_outputs=5

# RNN + SLP Model

# Define input layer
recurrent_input = Input(shape=(n_timesteps,n_features),name="TIMESERIES_INPUT")
static_input = Input(shape=(X_static.shape[1], ),name="STATIC_INPUT")


#RNN Layers
# layer - 1

rec_layer_one = Bidirectional(LSTM(128, kernel_regularizer=regularizers.l1_l2(l1=0, l2=0.01), recurrent_regularizer=regularizers.l1_l2(l1=0, l2=0.01),return_sequences=True),name ="BIDIRECTIONAL_LAYER_1")(recurrent_input)
rec_layer_one = Dropout(0.1,name ="DROPOUT_LAYER_1")(rec_layer_one)# layer - 2
rec_layer_two = Bidirectional(LSTM(64, kernel_regularizer=regularizers.l1_l2(l1=0, l2=0.01), recurrent_regularizer=regularizers.l1_l2(l1=0, l2=0.01)),name ="BIDIRECTIONAL_LAYER_2")(rec_layer_one)
rec_layer_two = Dropout(0.1,name ="DROPOUT_LAYER_2")(rec_layer_two)

# SLP Layers
static_layer_one = Dense(64,  kernel_regularizer=regularizers.l1_l2(l1=0, l2=0.01),  activation='relu',name="DENSE_LAYER_1")(static_input)

# Combine layers - RNN + SLP
combined = Concatenate(axis= 1,name = "CONCATENATED_TIMESERIES_STATIC")([rec_layer_two,static_layer_one])
combined_dense_two = Dense(64, activation='relu',name="DENSE_LAYER_2")(combined)
output = Dense(n_outputs,activation='sigmoid',name="OUTPUT_LAYER")(combined_dense_two)

# Compile ModeL
model = Model(inputs=[recurrent_input,static_input],outputs=[output])


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

                              
# binary cross entropy loss
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy',f1_m,precision_m, recall_m])


# focal loss

def focal_loss_custom(alpha, gamma):
  def binary_focal_loss(y_true, y_pred):
    fl = tfa.losses.SigmoidFocalCrossEntropy(alpha=alpha, gamma=gamma)
    y_true_K = K.ones_like(y_true)
    focal_loss = fl(y_true, y_pred)
    return focal_losS
  return binary_focal_lossmodel.compile(loss=focal_loss_custom(alpha=0.2, gamma=2.0), optimizer='adam', metrics=['accuracy',f1_m,precision_m, recall_m])

model.summary()


# fit network

history =  model.fit([np.asarray(x_train_reshape).astype('float32'), np.asarray(x_train_over_static).astype('float32')],y_train_reshape, epochs=epochs, batch_size=batch_size, verbose=verbose, validation_data=([np.asarray(x_val_reshape).astype('float32'), np.asarray(x_val_static).astype('float32')],y_val_reshape))# summarize history for accuracypyplot.plot(history.history['accuracy'])

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


#Save the model

from keras.models import model_from_json

# serialize classifier to JSON
classifier_json = model.to_json()
with open("rs_lstmp_classification.json", "w") as json_file:
    json_file.write(classifier_json)
# serialize weights to HDF5

model.save_weights("rs_lstmp_classification.tf",save_format='tf')

# classifier_json.save_weights("rs_lstmp_classification.h5")
# print("Saved classifier to disk")

#Now train from loaded model

# load json and create classifier
json_file = open('rs_lstmp_classification_33clusters.json', 'r')
loaded_classifier_json = json_file.read()
json_file.close()
loaded_classifier = model_from_json(loaded_classifier_json)
# load weights into new classifier
loaded_classifier.load_weights("rs_lstmp_classification_33clusters.tf")
print("Loaded classifier from disk")

#loading the model and checking accuracy on the test data
model = load_model('rs_lstmp_classification_33clusters.pkl')

from sklearn.metrics import accuracy_score
test_preds = model.predict_classes(X)
accuracy_score(y, test_preds)


