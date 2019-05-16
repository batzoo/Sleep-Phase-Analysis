import numpy as np
import utils
from utils import *


def save_array_npy(array,name,path = '..\\..\\numpy_files\\'):
	np.save(path+name,array)


def extract_data(database_folder,ind_signals = [1,2,15,16,17,18,3,19,22,4,14],subjects=[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]):

	for signal in range (len(ind_signals)) :
		data = []
		print('\n\nSIGNAL n° : ',signal+1,'/',len(ind_signals),'\n\n')
		for subject in subjects :
			print('SUBJECT n° : ',subject,'/',len(subjects))
			
			pathEDF = database_folder+'subject'+str(subject)+'.edf'
			pathHypnogram = database_folder+'HypnogramAASM_subject'+str(subject)+'.txt'

			raw_data_EDF = edfDataExtraction_interestingSignals(pathEDF,[ind_signals[signal]])
			raw_hypnogram = hypnogramDataExtraction(pathHypnogram)
			hypnogram = split_hypnogram(raw_hypnogram)
			temp = create_signal_label_arrays(raw_data_EDF,hypnogram)
			data.extend(temp)
		save_array_npy(data,utils.SIGNAL_LABELS[ind_signals[signal]])

def load_data(signal_name,numpy_files_folder = utils.NUMPY_FILES_FOLDER):
	data = np.load(numpy_files_folder+signal_name+'.npy')
	return data
