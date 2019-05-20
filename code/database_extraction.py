import numpy as np
import utils
from utils import *


def save_array_npy(array,name,temporal_mode,path = '..\\..\\numpy_files\\'):
	if(temporal_mode):
		np.save(path+'temporal\\'+name+'temporal',array)
	else:
		np.save(path+'frequency\\'+name+'frequency',array)

def extract_data(ind_signals = [1,2,15,16,17,18,3,19,22,4,14],database_folder = utils.DATABASE_FOLDER,subjects=[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]):
	print("Extract [T]emporal data or [F]requential")
	temporal_mode = True
	temp =  False
	while(temp == False):
		enter=input()
		if(enter=='T'):
			temporal_mode = True
			temp = True
		elif(enter=='F'):
			temporal_mode = False
			temp = True
		else:
			print("press T or F")
	for ind_signal in range (len(ind_signals)) :
		data = []
		print('\n\nSIGNAL n° : ',ind_signal+1,'/',len(ind_signals),'\n\n')
		for subject in subjects :
			print('SUBJECT n° : ',subject,'/',len(subjects))
			
			pathEDF = database_folder+'subject'+str(subject)+'.edf'
			pathHypnogram = database_folder+'HypnogramAASM_subject'+str(subject)+'.txt'

			raw_data_EDF = edfDataExtraction_interestingSignals_unique(pathEDF,ind_signals[ind_signal])
			raw_hypnogram = hypnogramDataExtraction(pathHypnogram)
			hypnogram = split_hypnogram(raw_hypnogram)

			signals = splitSignal_notime(30,raw_data_EDF)
			if(temporal_mode == False):
				signals_freq = []
				for signal in signals:
					signal_freq = spectrumCalculation_notime(signal)
					signal_freq = signal_freq[3000:3750]
					signal_freq = signalSmoother(signal_freq,10)
					signals_freq.append(signal_freq)
				signals = signals_freq

			data.extend(create_signal_label_arrays(signals,hypnogram))
		
		save_array_npy(data,utils.SIGNAL_LABELS[ind_signals[ind_signal]],temporal_mode)

def load_data(signal_name,numpy_files_folder = utils.NUMPY_FILES_FOLDER):
	data = np.load(numpy_files_folder+signal_name+'.npy')
	return data

extract_data()
