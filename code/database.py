import numpy as np
from utils import *
from database_extraction import *
import pyedflib

database_folder = utils.DATABASE_FOLDER

def resample(old_frequency,new_frequency,data):
	if(new_frequency > old_frequency):
		print("new frequency must be lower than old frequency")
		return
	else :
		new_data = []
		for i in range (len(data)):
			if ((i%(old_frequency/new_frequency))==0):
				new_data.append(data[i])

		return new_data
def decode_DREAMS_hypnogram(pathHypnogram):
	file = open(pathHypnogram+'.txt', "r")
	raw_hypnogram = file.read().split("\n")
	file.close()
	del raw_hypnogram[0]
	del raw_hypnogram[len(raw_hypnogram)-1]
	hypnogram = []
	n = (int)(30 / 5)
	for i in range (0,len(raw_hypnogram),n):
		if(i+n-1>len(raw_hypnogram)):
			break
		mean = np.mean(np.asarray((raw_hypnogram[i:i+n])).astype(np.int))
		hypnogram.append(mean.astype(np.int))
	return hypnogram


def decode_PHYSIONET_SLEEPEDFX_TELEMETRY_hypnogram(pathHypnogram):
	raw_hypnogram = pyedflib.EdfReader(pathHypnogram+'.edf').readAnnotations()
	temp = []
	hypnogram = []
	for i in range(len(raw_hypnogram[1])):
		for dur_30s in range((int)(raw_hypnogram[1][i]/30)):
			temp.append(raw_hypnogram[2][i])
	for i in temp:
		if i == 'Sleep stage 1':
			hypnogram.append(3) 
		elif i == 'Sleep stage 2':
			hypnogram.append(2) 
		elif i == 'Sleep stage 3':
			hypnogram.append(1) 
		elif i == 'Sleep stage W':
			hypnogram.append(5) 
		elif i == 'Sleep stage R':
			hypnogram.append(4) 
		elif i == 'Sleep stage 4':
			hypnogram.append(0)
	return hypnogram

def decode_hypnogram(pathHypnogram):
	if(utils.DATABASE == "DREAMS"):
		return(decode_DREAMS_hypnogram(pathHypnogram))
	elif(utils.DATABASE == "PHYSIONET_SLEEPEDFX_TELEMETRY"):
		return(decode_PHYSIONET_SLEEPEDFX_TELEMETRY_hypnogram(pathHypnogram))

def decode_PSG(pathPSG,ind_signal):
	filePSG = pyedflib.EdfReader(pathPSG)
	number_of_signals = filePSG.signals_in_file
	signal_buffer = filePSG.readSignal(ind_signal)
	signal = signal_buffer
	if(utils.DATABASE == "DREAMS"):
		signal = resample(200,100,signal_buffer)
	filePSG._close()
	return signal

def splitPSG(signal,frequency= utils.SAMPLING_FREQUENCY):
    splittedSignal=np.zeros((int(len(signal)/(30*frequency)),30*frequency))
    for i in range(int(len(signal)/(30*frequency))):
        for j in range(30*frequency):
            splittedSignal[i][j]=signal[i*30*frequency+j]
    return splittedSignal

def extract_data_freq(lissage,ind_signals = utils.INTERESTING_SIGNALS_INDS,subjects=np.arange(1,utils.NUMBER_SUBJECTS,dtype=np.int32)):
	for ind_signal in range(len(ind_signals)) :
		data = []
		print('\n\nSIGNAL n° : ',ind_signal+1,'/',len(ind_signals),'\n\n')
		for subject in subjects :
			print('SUBJECT n° : ',subject,'/',len(subjects))
			
			pathPSG = utils.DATABASE_FOLDER+'subject ('+str(subject)+').edf'
			pathHypnogram = utils.DATABASE_FOLDER+'Hypnogram ('+str(subject)+')'

			raw_data_PSG = decode_PSG(pathPSG,ind_signals[ind_signal])
			hypnogram = decode_hypnogram(pathHypnogram)

			signals = splitPSG(raw_data_PSG)
			signals_freq = []
			for signal in signals:
				signal_freq = spectrumCalculation_notime(signal)
				signal_freq = signal_freq[1500:1875]
				if(lissage>0):
					signal_freq = signalSmoother(signal_freq,lissage)
				signals_freq.append(signal_freq)
			signals = signals_freq

			data.extend(create_signal_label_maps(signals,hypnogram))
		
		save_array_npy(data,utils.SIGNAL_LABELS[ind_signals[ind_signal]]+str(lissage),False)

def extract_data_temp(ind_signals = utils.INTERESTING_SIGNALS_INDS,subjects=np.arange(1,utils.NUMBER_SUBJECTS+1)):
	
	for ind_signal in range(len(ind_signals)) :
		data = []
		print('\n\nSIGNAL n° : ',ind_signal+1,'/',len(ind_signals),'\n\n')
		for subject in subjects :
			print('SUBJECT n° : ',subject,'/',len(subjects))
			
			pathPSG = utils.DATABASE_FOLDER+'subject ('+str(subject)+').edf'
			pathHypnogram = utils.DATABASE_FOLDER+'Hypnogram ('+str(subject)+').edf'

			raw_data_PSG = decode_PSG(pathPSG,ind_signals[ind_signal])
			raw_hypnogram = decode_hypnogram(pathHypnogram)

			signals = splitPSG(raw_data_PSG)
			signals = signals_freq

			data.extend(create_signal_label_maps(signals,hypnogram))
		
		save_array_npy(data,utils.SIGNAL_LABELS[ind_signals[ind_signal]],False)

def create_signal_label_maps(PSG_signal,hypnogram):
	data=[]
	for i in range([len(hypnogram),len(PSG_signal)][np.argmin([len(hypnogram),len(PSG_signal)])]):
		data.append([PSG_signal[i],hypnogram[i]])
	return data

def load_data(signal_name,numpy_files_folder = utils.NUMPY_FILES_FOLDER):
	data = np.load(numpy_files_folder+signal_name+'.npy')
	return data
def save_array_npy(array,name,temporal_mode,path = utils.NUMPY_FILES_FOLDER):
	if(temporal_mode):
		np.save(path+name+'temporal',array)
	else:
		np.save(path+name+'frequency',array)

# extract_data_freq(0)
