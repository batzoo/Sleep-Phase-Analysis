from __future__ import absolute_import, division, print_function

import pathlib

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from utils import *
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

name_signaux_interessants = ['FP1-A2','CZ-A1','O1-A2','FP2-A1','O2-A1','CZ2-A1','EMG1','EMG2','EMG3','EOG1','EOG2']

subject = 1 
signal_index = 1 #indice signal voulu
database_folder = '..\\..\\Pologne\\Dataset\\'

pathEDF = database_folder+'subject'+str(subject)+'.edf'
pathHypnogram = database_folder+'HypnogramAASM_subject'+str(subject)+'.txt'

raw_data_EDF = edfDataExtraction_interestingSignals(pathEDF)

raw_hypnogram = hypnogramDataExtraction(pathHypnogram)

hypnogram = split_hypnogram(raw_hypnogram,5)

def build_model():
	model=keras.Sequential()
	model.add(layers.Dense(300,input_dim=6000))
	model.add(layers.Dense(200))
	model.add(layers.Dense(100))
	model.add(layers.Dense(50))
	model.add(layers.Dense(30))
	model.add(layers.Dense(1))

	optimizer=keras.optimizers.RMSprop(0.001)

	model.compile(loss='mean_squared_error',
                   optimizer=optimizer,
                   metrics=['mean_absolute_error', 'mean_squared_error'])
	return model

signal = raw_data_EDF[signal_index]
data = create_signal_label_arrays(signal,200,hypnogram)
training_data=[]
training_label=[]
test_data=[]
test_label=[]
print(data[2][0])
for i in range(700):
	training_data.append(data[i][0])
	training_label.append(data[i][1])


training_data=np.asarray(training_data)
training_label=np.asarray(training_label)
print(training_data.shape)


print("DATA" ,training_data)

print("LABEL" ,training_label)

model=build_model()

model.summary()

model.fit(training_data, training_label, epochs=40 ,verbose=1)
