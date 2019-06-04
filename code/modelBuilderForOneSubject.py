
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

import numpy as np

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers.embeddings import Embedding

from utils import *
from database_extraction import *

print("Loading datasets ...")
print("\n=============================================\n")
dataf=load_data("../../Datasets/separated_subjects/CZ2-A1frequency101dimsubject3")

labels=[]
features=[]
validation_labels=[]
validation_features=[]

print("Building the training and validation data ...")
print("\n=============================================\n")

for i in range(len(dataf)):
	if(len(features)<750 and (i%5)!=0):
		features.append(dataf[i,0])
		labels.append(dataf[i,1])
	else:
		validation_features.append(dataf[i,0])
		validation_labels.append(dataf[i,1])

labels = np.asarray(labels)
features = np.asarray(features)
validation_labels = np.asarray(validation_labels)
validation_features = np.asarray(validation_features)

print("\n=============================================\n")
print("Total number of training features = ",len(features)," and total number of training labels = ",len(labels))
print("Total number of validation features = ",len(validation_features)," and total number of validation labels = ",len(validation_labels))
print("\n=============================================\n")
print("Building the model ...")
print("\n=============================================\n")

#version précédentes de la gestion des labels et des features

"""labels=[]
features=[]
for i in range(len(dataf)):
	features.append(dataf[i,0])
	labels.append(dataf[i,1])
labels = np.asarray(labels)
features = np.asarray(features)"""


model = Sequential()
model.add(Embedding(1000, 128, input_length=375))
model.add(LSTM(70, activation='sigmoid', recurrent_activation='sigmoid', use_bias=True, kernel_initializer='glorot_uniform', recurrent_initializer='orthogonal', bias_initializer='zeros', unit_forget_bias=True, kernel_regularizer=None, recurrent_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, recurrent_constraint=None, bias_constraint=None, dropout=0.0, recurrent_dropout=0.0, implementation=1, return_sequences=False, return_state=False, go_backwards=False, stateful=False, unroll=False)
)
model.add(Dense(1, activation='relu'))
model.compile(loss='mean_absolute_error', 
              optimizer=tf.train.RMSPropOptimizer(0.01), 
              metrics=['accuracy'])
print("\n=============================================\n")
print(model.summary())
print("\n=============================================\n")
print("Beginning of the model training !")
print("\n=============================================\n")

model.fit(features, labels, epochs=20, batch_size = None)

print("\n=============================================\n")
print("End of the model training and saving it !")
print("\n=============================================\n")

model.save("modelSubject3.h5")

predictions=model.predict(features)
print("\n=============================================\n")
print("Predictions = ",predictions)
print("\n=============================================\n")
counterN1=0
counterN2=0
for i in range(len(predictions)):
    if(round(predictions[i][0])==labels[i]):
        counterN1+=1
    else:
        counterN2+=1

print("\n=============================================\n")
print("TRUE : ",counterN1,"FALSE : ",counterN2)
print("TRUE : ",counterN1*100/len(predictions),"%, FALSE : ",counterN2*100/len(predictions),"%")
print("\n\n\nEND of the model building !")