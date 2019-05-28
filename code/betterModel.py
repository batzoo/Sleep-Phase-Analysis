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
	model.add(layers.Dense(375,input_dim=375))
	model.add(layers.Activation('sigmoid'))
	model.add(layers.Dense(200))
	model.add(layers.Activation('sigmoid'))
	model.add(layers.Dense(100))
	model.add(layers.Activation('relu'))
	model.add(layers.Dense(1,activation='relu'))
	


	optimizer = keras.optimizers.Adam(lr=0.0001,beta_1=0.99, beta_2=0.999, epsilon=0.01, decay=0.01, amsgrad=False)

	model.compile(loss='mean_absolute_error',
                   optimizer=optimizer,
                   metrics=['acc'])
	model.summary()
	return model

def build_model_LSTM():
	model=keras.Sequential()
	model.add(layers.LSTM(128,input_shape=(2,375),return_sequences=True,activation='sigmoid'))
	model.add(layers.LSTM(128,input_shape=(128,),return_sequences=True,activation='tanh'))
	model.add(layers.LSTM(64,input_shape=(128,),return_sequences=False,activation='tanh'))
	model.add(layers.Dense(64,activation='tanh'))
	model.add(layers.Dense(32,activation='tanh'))
	model.add(layers.Dense(1,activation='relu'))
	
	optimizer = keras.optimizers.Adam(lr=0.001)

	model.compile(loss='mean_absolute_error',
                   optimizer=optimizer,
                   metrics=['acc'])
	model.summary()
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
	return training_data,training_label,test_data,test_label

def prepareDataForLSTM(signal1,signal2,percentage_training):
	data=load_data(signal1+"frequency")
	data2=load_data(signal2+"frequency")
	training_data=[]
	test_data=[]
	training_label=[]
	test_label=[]
	for i in range(int(len(data)/100*percentage_training)):
		training_data.append(np.asarray([data[i][0],data2[i][0]]))
		training_label.append(np.asarray(data[i][1]))
	
	for j in range(int(len(data)/100*percentage_training)+1,len(data)):
		test_data.append(np.asarray([data[j][0],data2[j][0]]))
		test_label.append(np.asarray(data[j][1]))

	training_data=np.asarray(training_data)
	training_label=np.asarray(training_label)
	test_data=np.asarray(test_data)
	test_label=np.asarray(test_label)

	return training_data,training_label,test_data,test_label

def prepareData(model_type):
	ctr=0
	print("Enter your signal(s) : ")
	for i in utils.INTERESTING_SIGNALS_INDS:
		print(utils.SIGNAL_LABELS[i],ctr)
		ctr+=1
	if(model_type=='L'):
		index_1=int(input("Signal 1 : "))
		index_2=int(input("Signal 2 : "))	
		training_data,training_label,test_data,test_label=prepareDataForLSTM(utils.SIGNAL_LABELS[index_1],utils.SIGNAL_LABELS[index_2],80)
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
	data=load_data("FP1-A2frequency")
	data2=load_data("CZ-A1frequency")
	data3=load_data("FP2-A1frequency")
	data4=load_data("CZ2-A1frequency")
	data5=load_data("O1-A2frequency")
	data6=load_data("O2-A1frequency")
	training_data=[]
	test_data=[]
	training_label=[]
	test_label=[]
	for i in range(int(len(data)/100*percentage_training)):
		training_data.append(data[i][0])
		training_label.append(data[i][1])
	for k in range(int(len(data)/100*percentage_training)):
		training_data.append(data2[k][0])
		training_label.append(data2[k][1])
	for l in range(int(len(data)/100*percentage_training)):
		training_data.append(data3[l][0])
		training_label.append(data3[l][1])
	for m in range(int(len(data)/100*percentage_training)):
		training_data.append(data4[m][0])
		training_label.append(data4[m][1])
	for n in range(int(len(data)/100*percentage_training)):
		training_data.append(data5[n][0])
		training_label.append(data5[n][1])
	for o in range(int(len(data)/100*percentage_training)):
		training_data.append(data6[o][0])
		training_label.append(data6[o][1])

	for p  in range(int(len(data)/100*percentage_training+1),len(data)):	
		test_data.append(data[p][0])
		test_label.append(data[p][1])
	for r in range(int(len(data)/100*percentage_training+1),len(data)):	
		test_data.append(data2[r][0])
		test_label.append(data2[r][1])
	for s in range(int(len(data)/100*percentage_training+1),len(data)):	
		test_data.append(data3[s][0])
		test_label.append(data3[s][1])
	for t in range(int(len(data)/100*percentage_training+1),len(data)):	
		test_data.append(data4[t][0])
		test_label.append(data4[t][1])
	for u in range(int(len(data)/100*percentage_training+1),len(data)):	
		test_data.append(data5[u][0])
		test_label.append(data5[u][1])
	for v in range(int(len(data)/100*percentage_training+1),len(data)):	
		test_data.append(data6[v][0])
		test_label.append(data6[v][1])

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
	#training_data,training_label,test_data,test_label=prepareData(model_type)
	
	training_data,training_label,test_data,test_label=load_all_data(80)
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