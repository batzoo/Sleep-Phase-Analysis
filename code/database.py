import numpy as np
from utils import *
from database_extraction import *
import pyedflib

def spectrumCalculation(signal,dim2,samplingFrequency=200):
	"""
	Input:  - signal
	
	Output: - Return a list which is [[spectrumList],[frequencyList]]
	"""
	Nfft = len(signal) #Number of dots for the fft
	
	#Spectrum calculation
	S = spfft.fft(signal, n=Nfft)
	spectrum = abs(spfft.fftshift(S))
	    
	#Frequency calculation
	frequency = np.arange(-samplingFrequency/2,samplingFrequency/2,samplingFrequency/Nfft) #All the frequency dots
	if(dim2):
		return spectrum,frequency
	else:
		return spectrum

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
	if(utils.DATABASE == "DREAMS"):
		signal = resample(200,100,signal_buffer)
	else:
		signal = signal_buffer	
	filePSG._close()
	return signal

def splitPSG(signal,dim2,frequency= utils.SAMPLING_FREQUENCY):
	if(dim2):
		splittedSignal=np.zeros((int(len(signal)/(30*frequency)),2,30*frequency))
	else:
		splittedSignal=np.zeros((int(len(signal)/(30*frequency)),30*frequency))
	for i in range(int(len(signal)/(30*frequency))):
		for j in range(30*frequency):
			if(dim2):
				splittedSignal[i][0][j]=signal[i*30*frequency+j]
				splittedSignal[i][1][j]=j/frequency
			else:
				splittedSignal[i][j]=signal[i*30*frequency+j]
	return splittedSignal

def extract_data_freq(separate_subject,lissage,dim2,ind_signals = utils.INTERESTING_SIGNALS_INDS,subjects=np.arange(1,utils.NUMBER_SUBJECTS,dtype=np.int32)):

	for ind_signal in range(len(ind_signals)) :
		data = []
		print('\n\nSIGNAL n° : ',ind_signal+1,'/',len(ind_signals),'\n\n')
		for subject in subjects :
			if(separate_subject):
				data = []
			print('SUBJECT n° : ',subject,'/',len(subjects))
			
			pathPSG = utils.DATABASE_FOLDER+'subject ('+str(subject)+').edf'
			pathHypnogram = utils.DATABASE_FOLDER+'Hypnogram ('+str(subject)+')'

			raw_data_PSG = decode_PSG(pathPSG,ind_signals[ind_signal])
			hypnogram = decode_hypnogram(pathHypnogram)
			signals = splitPSG(raw_data_PSG,dim2)
			signals_freq = []
			for signal in signals:
				if(dim2):
					signal_freq=[]
					signal_temp, frequency_temp = spectrumCalculation(signal[0],dim2)
					if(lissage>0):
						signal_temp = signalSmoother(signal_temp,lissage)
						signal_freq.extend([signal_temp[1500:1875],frequency_temp[1500:1875]])
					else:
						signal_freq.extend([signal_temp[1500:1875],frequency_temp[1500:1875]])
				else:
					signal_freq = spectrumCalculation(signal,dim2)
					signal_freq = signal_freq[1500:1875]
					if(lissage>0):
						signal_freq = signalSmoother(signal_freq,lissage)
				signals_freq.append(signal_freq)
			signals = signals_freq
			data.extend(create_signal_label_maps(signals,hypnogram,dim2))
			if(separate_subject):
				save_array_npy(data,utils.SIGNAL_LABELS[ind_signals[ind_signal]],False,lissage,dim2,str(subject))
		if(not(separate_subject)):
			save_array_npy(data,utils.SIGNAL_LABELS[ind_signals[ind_signal]],False,lissage,dim2)

def extract_data_temp(separate_subject,dim2,ind_signals = utils.INTERESTING_SIGNALS_INDS,subjects=np.arange(1,utils.NUMBER_SUBJECTS,dtype=np.int32)):
	
	for ind_signal in range(len(ind_signals)) :
		data = []
		print('\n\nSIGNAL n° : ',ind_signal+1,'/',len(ind_signals),'\n\n')
		for subject in subjects :
			if(separate_subject):
				data = []
			print('SUBJECT n° : ',subject,'/',len(subjects))
			
			pathPSG = utils.DATABASE_FOLDER+'subject ('+str(subject)+').edf'
			pathHypnogram = utils.DATABASE_FOLDER+'Hypnogram ('+str(subject)+')'

			raw_data_PSG = decode_PSG(pathPSG,ind_signals[ind_signal])
			hypnogram = decode_hypnogram(pathHypnogram)

			signals = splitPSG(raw_data_PSG,dim2)
		
			data.extend(create_signal_label_maps(signals,hypnogram,dim2))

			if(separate_subject):
				save_array_npy(data,utils.SIGNAL_LABELS[ind_signals[ind_signal]],True,0,dim2,str(subject))
		
		if(not(separate_subject)):
			save_array_npy(data,utils.SIGNAL_LABELS[ind_signals[ind_signal]],True,0,dim2)


def create_signal_label_maps(PSG_signal,hypnogram,dim2):
	data=[]
	for i in range([len(hypnogram),len(PSG_signal)][np.argmin([len(hypnogram),len(PSG_signal)])]):
		if(dim2):
			data.append([PSG_signal[i][0],PSG_signal[i][1],hypnogram[i]])
		else:
			data.append([PSG_signal[i],hypnogram[i]])
	return data

def load_data(signal_name,numpy_files_folder = utils.NUMPY_FILES_FOLDER):
	data = np.load(numpy_files_folder+signal_name+'.npy')
	return data
def save_array_npy(array,name,temporal_mode,lissage,dim2,subject='',path = utils.NUMPY_FILES_FOLDER):
	if(dim2):
		dim2 = "2dim"
	else:
		dim2 = "1dim"
	if(temporal_mode):
		if(subject != ''):
			np.save(path+'\\'+dim2+'\\separated_subjects\\'+name+'temporal'+str(lissage)+dim2+'subject'+str(subject),array)
		else:
			np.save(path+'\\'+dim2+'\\'+name+'temporal'+str(lissage)+dim2,array)
	else:
		if(subject != ''):
			np.save(path+'\\'+dim2+'\\separated_subjects\\'+name+'frequency'+str(lissage)+dim2+'subject'+str(subject),array)
		else:
			np.save(path+'\\'+dim2+'\\'+name+'frequency'+str(lissage)+dim2,array)

def main():
	extract_mode=""
	while( extract_mode!='F' and  extract_mode!='F' and extract_mode!='T' and  extract_mode!='t'):
		extract_mode=input("Extract in (F)requency or (T)emporal Domain \n").upper()
	if(extract_mode == 'F'):
		lissage = -1 
		while(lissage < 0 or lissage > 50):
			lissage=(int)(input("Choose smoothing degree (integer) : \n"))
	dimensions = ''
	while(dimensions !='Y' and dimensions != 'N'):
		dimensions = input("Extract with temporal/frequency array ? (Y)es / (N)o \n").upper()
	if(dimensions == 'Y'):
		dimensions = True
	elif(dimensions == 'N'):
		dimensions = False
	separate = ''
	while(separate != 'Y' and separate != 'N'):
		separate = input("Separate the subjects ? (Y)es/(N)o \n").upper()
	if(separate=='N'):
		separate = False
	elif(separate == 'Y'):
		separate = True
	if(extract_mode == 'F'):
		extract_data_freq(separate,lissage,dimensions)
	elif(extract_mode == 'T'):
		extract_data_temp(separate,dimensions)

if __name__ == "__main__":
    main()