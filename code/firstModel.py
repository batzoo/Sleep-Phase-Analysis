from __future__ import absolute_import, division, print_function

import pathlib

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from utils import *
from database_extraction import *
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

name_signaux_interessants = ['FP1-A2','CZ-A1','O1-A2','FP2-A1','O2-A1','CZ2-A1','EMG1','EMG2','EMG3','EOG1','EOG2']

NUMBER_OF_EPOCHS=50
model_number=""

def build_model():
	model=keras.Sequential()
	model.add(layers.Dense(30,input_dim=6000))
	model.add(layers.Activation('relu'))
	model.add(layers.Dense(375))
	model.add(layers.Activation('relu'))
	model.add(layers.Dense(1))

	optimizer=keras.optimizers.RMSprop(0.001)

	model.compile(loss='mean_absolute_error',
                   optimizer=optimizer,
                   metrics=['binary_accuracy'])
	return model
def load_model_ez():
	global model_number
	model_number=input("Entrer le numéro du modèle : ")
	loaded_model=keras.models.load_model("..\\..\\Pologne\\Models\\my_model",str(model_number),".h5")
	return loaded_model


def save_model(model):
	global model_number
	if(model_number==""):
		model.save("..\\..\\Pologne\\Models\\my_model4.h5")
	print("Saved as my_model2")

def train_model(model,training_data,training_label):
	global NUMBER_OF_EPOCHS
	model.fit(training_data, training_label, epochs=NUMBER_OF_EPOCHS ,verbose=1)
	save_model(model)	


def prepareData():
	data=load_data("FP1-A2")
	training_data=[]
	training_label=[]
	test_data=[]
	test_label=[]
	print(len(data))
	for i in range(16000):
		training_data.append(data[i][0])
		training_label.append(data[i][1])
	
	for j in range(16001,len(data)):
		test_data.append(data[j][0])
		test_label.append(data[j][1])
	
	training_data=np.asarray(training_data)
	training_label=np.asarray(training_label)
	test_data=np.asarray(test_data)
	test_label=np.asarray(test_label)
	return training_data,training_label,test_data,test_label
def master():
	global NUMBER_OF_EPOCHS
	ctr=0
	ctr2=0
	training_data,training_label,test_data,test_label=prepareData()


	print("Nouveau ou ancien modele : (O)ld/(N)ew ")
	enter=input()

	if(enter=='O'):
		model=load_model_ez()
	elif(enter=='N'):
		model=build_model()
	else:
		print("O ou N")
		return


	print("Train ? (O)ui/(N)on")
	enter2=input()

	if(enter2=='O'):
		NUMBER_OF_EPOCHS=int(input("Number of Epochs :"))
		train_model(model,training_data,training_label)
	
	predictions=model.predict(test_data)
	for i in range(len(predictions)):
		if(round(predictions[i][0])==test_label[i]):
			ctr+=1
		else:
			ctr2+=1
	print("TRUE : ",ctr,"FALSE : ",ctr2)	

master()


#signal = raw_data_EDF[signal_index]
#data = create_signal_label_arrays(signal,200,hypnogram)
#training_data=[]
#training_label=[]
#test_data=[]
#test_label=[]
#print(data[2][0])
#for i in range(700):
#	training_data.append(data[i][0])
#	training_label.append(data[i][1])
#
#for j in range(701,len(data)):
#	test_data.append(data[j][0])
#	test_label.append(data[j][1])
#
#training_data=np.asarray(training_data)
#training_label=np.asarray(training_label)
#test_data=np.asarray(test_data)
#test_label=np.asarray(test_label)
#
#print("DATA" ,training_data)
#
#print("LABEL" ,training_label)
#
#
#model=load_model_ez()
#
#
#print(test_data[0])
#ctr=0
#ctr2=0
#predictions=model.predict(test_data)
#for z in range(len(predictions)):
#	if(round(predictions[z][0])==test_label[z]):
#		ctr+=1
#	else:
#		ctr2+=1
#print("TRUE : ",ctr,"FALSE : ",ctr2)