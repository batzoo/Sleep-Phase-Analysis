from __future__ import absolute_import, division, print_function

import pathlib
from tensorflow.keras.utils import plot_model

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from utils import *
from database_extraction import *
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import optimizers
name_signaux_interessants = ['FP1-A2','CZ-A1','O1-A2','FP2-A1','O2-A1','CZ2-A1','EMG1','EMG2','EMG3','EOG1','EOG2']

NUMBER_OF_EPOCHS=50
model_number=""
models_file="..\\..\\Pologne\\Models\\"
loaded=False


def compile_model(model):


	model.compile(loss="mean_absolute_error",optimizer=optimizers.Adam(),
                   metrics=['binary_accuracy'])
	return model

def build_model(model_type):
	if(model_type=='L'):
		model=build_model_LSTM()
	elif(model_type=='D'):
		model=build_model_Dense()
	else:
		model_number=int(input("Model number ? "))
		model=load_model_from_file(model_number)
		model_type=input("Is it (D)ense or (L)STM ?")
	return model,model_type
def build_model_Dense():
	model=keras.Sequential()
	model.add(layers.Dense(20,input_dim=2))
	model.add(layers.Activation('sigmoid'))
	model.add(layers.Dense(200,activation='relu'))
	model.add(layers.Dense(400,activation='relu'))
	model.add(layers.Dense(100,activation='relu'))
	model.add(layers.Dense(1,activation='softplus'))
	


	optimizer = keras.optimizers.Adam(lr=0.0001)

	model.compile(loss='mean_absolute_error',
                   optimizer=optimizer,
                   metrics=['acc'])
	model.summary()
	return model

def build_model_LSTM():
	model=keras.Sequential()
	layer=layers.Conv1D(32,4,input_shape=(2,375),activation='relu',padding='same')
	model.add(layer)
	print("input conv 8",layer.input_shape)
	print(layer.output_shape)
	layer=layers.MaxPooling1D(1)
	model.add(layer)
	print("input maxpool 1",layer.input_shape)
	print(layer.output_shape)
	layer=layers.Conv1D(16,8,activation='relu',padding='same')
	model.add(layer)
	print("input conv 16",layer.input_shape)
	print(layer.output_shape)
	layer=layers.MaxPooling1D(2)
	model.add(layer)
	print("input maxpool 1",layer.input_shape)
	print(layer.output_shape)
	layer=layers.Conv1D(8,8,activation='relu',padding='same')
	model.add(layer)
	print("input conv 32",layer.input_shape)
	print(layer.output_shape)
	layer=layers.MaxPooling1D(1)
	model.add(layer)
	print("input maxpool 1",layer.input_shape)
	print(layer.output_shape)
	model.add(layers.LSTM(25,input_shape=(375,),return_sequences=True,activation='relu'))
	model.add(layers.LSTM(25,input_shape=(25,),return_sequences=False,activation='relu'))
	model.add(layers.Dense(32,activation='relu'))
	model.add(layers.Dense(1,activation='softplus'))
	
	optimizer = keras.optimizers.RMSprop(lr=0.0005)

	model.compile(loss='mean_absolute_error',
                   optimizer=optimizer,
                   metrics=['acc'])
	return model

def load_model_from_file(model_number):
	loaded=True
	print("loading my_model"+str(model_number)+".h5")

	loaded_model=keras.models.load_model(models_file+"my_model"+str(model_number)+".h5")
	
	print("my_model"+str(model_number)+" loaded ")
	loaded_model.summary()
	return loaded_model


def save_model(model,model_number):
	print("Saving model")
	model.save(models_file+"my_model"+str(model_number)+".h5")
	print("Saved as my_model"+str(model_number))

def train_model(model,training_data,training_label):
	NUMBER_OF_EPOCHS=int(input("Number of epochs : "))

	history=model.fit(training_data, training_label, epochs=NUMBER_OF_EPOCHS ,verbose=1)
	save=""
	train_again=""
	while(save!='Y'and save!='N' and save!='n' and save!='y'):
		save=input("Save model ? (Y)es/(N)o \n ").upper()
	if(save=='Y'):
		model_number=int(input("Model number ? (already existing number will overwrite the current model) "))
		save_model(model,model_number)
	while(train_again!='Y' and train_again!='N' and train_again!='n' and train_again!='y'):
		train_again=input("Train again ? (Y)es/(N)o \n").upper()
	if(train_again=='Y'):
		train_model(model,training_data,training_label)
	return history



def prepareDataForDense(signal,percentage_training):
	data=load_data(signal+"frequency")
	training_data=[]
	test_data=[]
	training_label=[] 
	test_label=[]
	for i in range(int(len(data)/100*percentage_training)):
		training_data.append(data[i][0])
		training_label.append(data[i][1])
	
	for j in range(int(len(data)/100*percentage_training+1),len(data)):	
		test_data.append(data[j][0])
		test_label.append(data[j][1])

	training_data=np.asarray(training_data)
	training_label=np.asarray(training_label)
	test_data=np.asarray(test_data)
	test_label=np.asarray(test_label)

	print(training_data[0][0])
	print(training_data[0][1])
	return training_data,training_label,test_data,test_label

def prepareDataForLSTM(signal1,percentage_training):
	data=load_data(signal1+"frequency10")
	training_data=[]
	test_data=[]
	training_label=[]
	test_label=[]

	for i in range(int(len(data)/100*percentage_training)):
		if(not(data[i][0][np.argmax(data[i][0])]>11000)):
			training_data.append(np.asarray([data[i][0],data[i][1]]))
			training_label.append(np.asarray(data[i][2]))
	
	for j in range(int(len(data)/100*percentage_training)+1,len(data)):
		test_data.append(np.asarray([data[j][0],data[j][1]]))
		test_label.append(np.asarray(data[j][2]))
	
	training_data=np.asarray(training_data)
	training_label=np.asarray(training_label)
	test_data=np.asarray(test_data)
	test_label=np.asarray(test_label)
	print(training_data[0][0])
	print(training_data[0][1])
	return training_data,training_label,test_data,test_label

def prepareData(model_type):
	ctr=0
	print("Enter your signal(s) : ")
	for i in utils.INTERESTING_SIGNALS_INDS:
		print(utils.SIGNAL_LABELS[i],ctr)
		ctr+=1
	if(model_type=='L'):
		index_1=int(input("Signal 1 : "))
		training_data,training_label,test_data,test_label=prepareDataForLSTM(utils.SIGNAL_LABELS[index_1],80)
	elif(model_type=='D'):
		index=int(input("Signal : "))
		training_data,training_label,test_data,test_label=prepareDataForDense(utils.SIGNAL_LABELS[index],80)
	print("Shape : ",training_data.shape)
	return training_data,training_label,test_data,test_label

def verification(model,test_data,test_label):
	predictions=model.predict(test_data)
	right=0
	wrong=0
	for i in range(len(predictions)):
		if(round(predictions[i][0])==test_label[i]):
			right+=1
		else:
			print(round(predictions[i][0]),test_label[i])
			wrong+=1
	print("TRUE : ",right,"FALSE : ",wrong)

def load_all_data(percentage_training):
	data=load_data("FP1-A2frequency10")
	data2=load_data("CZ-A1frequency10")
	data3=load_data("FP2-A1frequency10")
	data4=load_data("CZ2-A1frequency10")
	data5=load_data("O1-A2frequency10")
	data6=load_data("O2-A1frequency10")
	training_data=[]
	test_data=[]
	training_label=[]
	test_label=[]
	for i in range(int(len(data)/100*percentage_training)):
		training_data.append([data[i][0],data[i][1]])
		training_label.append(data[i][2])
	for k in range(int(len(data)/100*percentage_training)):
		training_data.append([data2[k][0],data2[k][1]])
		training_label.append(data2[k][2])
	for l in range(int(len(data)/100*percentage_training)):
		training_data.append([data3[l][0],data3[l][1]])
		training_label.append(data3[l][2])
	for m in range(int(len(data)/100*percentage_training)):
		training_data.append([data4[m][0],data4[m][1]])
		training_label.append(data4[m][2])
	for n in range(int(len(data)/100*percentage_training)):
		training_data.append([data5[n][0],data5[n][1]])
		training_label.append(data5[n][2])
	for o in range(int(len(data)/100*percentage_training)):
		training_data.append([data6[o][0],data6[o][1]])
		training_label.append(data6[o][2])

	for p  in range(int(len(data)/100*percentage_training+1),len(data)):	
		test_data.append([data[p][0],data[p][1]])
		test_label.append(data[p][2])
	for r in range(int(len(data)/100*percentage_training+1),len(data)):	
		test_data.append([data2[r][0],data2[r][1]])
		test_label.append(data2[r][2])
	for s in range(int(len(data)/100*percentage_training+1),len(data)):	
		test_data.append([data3[s][0],data3[s][1]])
		test_label.append(data3[s][2])
	for t in range(int(len(data)/100*percentage_training+1),len(data)):	
		test_data.append([data4[t][0],data4[t][1]])
		test_label.append(data4[t][2])
	for u in range(int(len(data)/100*percentage_training+1),len(data)):	
		test_data.append([data5[u][0],data5[u][1]])
		test_label.append(data5[u][2])
	for v in range(int(len(data)/100*percentage_training+1),len(data)):	
		test_data.append([data6[v][0],data6[v][1]])
		test_label.append(data6[v][2])

	training_data=np.asarray(training_data)
	training_label=np.asarray(training_label)
	test_data=np.asarray(test_data)
	test_label=np.asarray(test_label)
	return training_data,training_label,test_data,test_label

def main():
	model_type=""
	while( model_type!='L' and  model_type!='l' and model_type!='D' and  model_type!='d' and model_type!='O'):
		model_type=input("(L)STM or (D)ense or load (O)ld model ? \n").upper()
	model,model_type=build_model(model_type)
	training_data,training_label,test_data,test_label=prepareData(model_type)
	
	#training_data,training_label,test_data,test_label=load_all_data(80)
	history=train_model(model,training_data,training_label)

	# Plot training & validation accuracy values
	plt.plot(history.history['acc'])
	plt.plot(history.history['loss'])
	plt.title('Model accuracy')
	plt.ylabel('Accuracy')
	plt.xlabel('Epoch')
	plt.legend(['Train', 'Test'], loc='upper left')
	plt.show()
	verification(model,test_data,test_label)
main()