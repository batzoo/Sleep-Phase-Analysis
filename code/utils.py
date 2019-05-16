import pyedflib
import numpy as np
import matplotlib.pyplot as plt

from scipy import fftpack as spfft
from scipy import signal as spsig


SIGNAL_LABELS = ['ECG', 'FP1-A2', 'CZ-A1', 'EMG1', 'EOG1', 'VTH', 'VAB', 'NAF2P-A1', 'NAF1', 'PHONO', 'PR', 'SAO2', 'PCPAP', 'POS', 'EOG2', 'O1-A2', 'FP2-A1', 'O2-A1', 'CZ2-A1', 'EMG2', 'PULSE', 'VTOT', 'EMG3']
INTERESTING_SIGNALS_LABELS = ['FP1-A2','CZ-A1','O1-A2','FP2-A1','O2-A1','CZ2-A1','EMG1','EMG2','EMG3','EOG1','EOG2']
DATABASE_FOLDER = '..\\..\\Dataset\\'
NUMPY_FILES_FOLDER = '..\\..\\numpy_files\\'

def partialSum(signal, index, gap):
    """
    Input:  - signal (table with the data of the signal)
            - index (the middle of the partial sum)
            - gap (the gap you want before and after the index)
            
    Ouput:  - Return the result of the partial sum between [index-gap,index+gap]
    """
    sumResult=0
    for i in range(-gap,gap):
        sumResult+=signal[index+i]
    return sumResult

def partialAverage(signal, index, gap):
    """
    Input:  - signal (table with the data of the signal)
            - index (the middle of the partial sum)
            - gap (the gap you want before and after the index)
    
    Output: - Return the result of the partial average between [index-gap,index+gap]
    """
    partialAverage=partialSum(signal, index, gap)/(2*gap+1)
    return partialAverage



def signalSmoother(signal, gap):
    """
    Input:  - signal (table with the data of the signal)
            - gap (the gap you want before and after the index)
    
    Output: - Return a list with the smoothing signal
    """
    signalPrime=signal[:]
    for k in range(gap,len(signal)-gap):
        signalPrime[k]=partialAverage(signal, k, gap)
    return np.asarray(signalPrime)

def signalIntervalExtractionFromBuffer(buffer,signalIndex=0,intervalMin=0,intervalMax=100,samplingFrequency=200):
    """
    Input:  - buffer
            - signalIndex=0
            - samplingFrequency=200
            - intervalMin=0
            - intervalMax=100
    
    Output: - Return a list which is [[signalList],[timeList]]
    """
    
    #Loading of the signal CZ-A1 from the buffer
    signal=buffer[2,intervalMin:intervalMax]

    #Calculation of the time
    numberOfSecond = (intervalMax-intervalMin)/samplingFrequency #Time in the interval
    time = np.arange(0,numberOfSecond,1/samplingFrequency) #Time for ploting
    
    return np.array([signal,time])

def spectrumCalculation(signal,samplingFrequency=200):
    """
    Input:  - signal
    
    Output: - Return a list which is [[spectrumList],[frequencyList]]
    """
    Nfft = len(signal[0]) #Number of dots for the fft
    
    #Spectrum calculation
    S = spfft.fft(signal[0], n=Nfft)
    spectrum = abs(spfft.fftshift(S))
        
    #Frequency calculation
    frequency = np.arange(-samplingFrequency/2,samplingFrequency/2,samplingFrequency/Nfft) #All the frequency dots

    return np.array([spectrum,frequency])

def spectrumCalculation_signalunique(signal,samplingFrequency=200):
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

    return np.array([spectrum,frequency])

def displaySignal(signal):
    #Diplay the signal
    plt.figure(figsize=(12,8))
    plt.title('Signal:')
    plt.plot(signal[1],signal[0],'r', linewidth = 3)
    plt.xlabel('Time (s)')
    plt.ylabel('Level (µV)')
    plt.tight_layout()
    plt.show()

def displaySignal_bis(signal,frequency):
    #Diplay the signal

    t = np.arange(0,len(signal)/200,1/200)
    plt.figure(figsize=(12,8))
    plt.title('Signal:')
    plt.plot(t,signal,'r', linewidth = 3)
    plt.xlabel('Time (s)')
    plt.ylabel('Level (µV)')
    plt.tight_layout()
    plt.show()

def displaySpectrum(spectrum,samplingFrequency=200):
    #Diplay the spectrum
    plt.figure(figsize=(12,8))
    plt.title('Linear spectrum:')
    plt.plot(spectrum[1],spectrum[0],'r', linewidth = 3)
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Level')
    plt.xlim([0,samplingFrequency/2])
    plt.tight_layout()


def hypnogramIntervalsCalculation(linesList):
    """
    Input:  - linesList (table of data which should come from hypnogramDataExtraction)
    
    Output: - Return the list of all the interval
    """
    #initialisation of the values
    buffer = [0,0,0]
    intervalsList =[]
    buffer[0] = int(linesList[0])
    buffer[1] = 0
    #Determination of the intervals
    for i in range (1,len(linesList)-1):
        if buffer[0]!=int(linesList[i]):
            buffer[2] = i-1
            intervalsList.append(buffer[:])
            buffer[0] = int(linesList[i])
            buffer[1] = i
    return np.asarray(intervalsList)


def edfDataExtraction(filePath):
    """
    Input:  - filePath
    
    Output: - Return a buffer (a matrix) with all the signals inside
    """
    fileEDF = pyedflib.EdfReader(filePath) #Openning the EDF file
    number_of_signals = fileEDF.signals_in_file
    signal_labels = fileEDF.getSignalLabels()
    signal_buffer = np.zeros((number_of_signals,fileEDF.getNSamples()[0]))
    
    #Display some information about the edf file
    print("Number of signals: ",number_of_signals)
    print("\nSignals labels: ",signal_labels)
    
    #Storing the data in a buffer
    for i in np.arange(number_of_signals):
        signal_buffer[i, :] = fileEDF.readSignal(i)
        
    #Display some information about the buffer
    print("Signals buffer:\n   Size line =",len(signal_buffer),"\n   Size column =",len(signal_buffer[0]))

    fileEDF._close() #Closing the EDF file
    return signal_buffer

def edfDataExtraction_interestingSignals(filePath, signals_index = [1,2,15,16,17,18,3,19,22,4,14]):
    """
    Input:  - filePath
    
    Output: - Return a buffer (a matrix) with all the signals inside
    """
    
    
    fileEDF = pyedflib.EdfReader(filePath) #Openning the EDF file
    n=len(signals_index)

    signal_buffer = np.zeros((n,fileEDF.getNSamples()[0]))
	
    signal_labels = fileEDF.getSignalLabels()
	#Display some information about the edf file
	# print("Number of signals: ",len(signals_index))
	# print("\nSignals labels: ",signal_labels)

	
	#Storing the data in a buffer
    for i in range (n):
        signal_buffer[i, :] = fileEDF.readSignal(signals_index[i])
        
    fileEDF._close() #Closing the EDF file
    return signal_buffer

def edfDataExtraction_interestingSignals_unique(filePath, signal_index):
    """
    Input:  - filePath
    
    Output: - Return a buffer (a matrix) with all the signals inside
    """
    
    
    fileEDF = pyedflib.EdfReader(filePath) #Openning the EDF file

    signal_buffer = np.zeros((1,fileEDF.getNSamples()[0]))
    
    signal_labels = fileEDF.getSignalLabels()
    #Display some information about the edf file
    # print("Number of signals: ",len(signals_index))
    # print("\nSignals labels: ",signal_labels)

    
    #Storing the data in a buffer
    
    signal_buffer = fileEDF.readSignal(signal_index)

    fileEDF._close() #Closing the EDF file
    return signal_buffer

def hypnogramDataExtraction(filePath):
    """
    Input:  - filePath
    
    Output: - Return a list with all the data
    """
    file = open(filePath, "r")
    lines = file.read().split("\n")
    file.close()
    del lines[0]
    del lines[len(lines)-1]
    return np.asarray(lines)

def splitSignal(interval,signal):
    splittedSignal=np.zeros((int(len(signal)/(interval*200)),2,interval*200))
    # print("splitted",splittedSignal.shape,"oui",splittedSignal[0])
    for i in range(int(len(signal)/(interval*200))):
        for j in range(interval*200):
            splittedSignal[i][0][j]=signal[i*interval*200+j]
            splittedSignal[i][1][j]=j/200
    return splittedSignal

def splitSignal_notime(interval,signal):
    splittedSignal=np.zeros((int(len(signal)/(interval*200)),interval*200))
    # print("splitted",splittedSignal.shape,"oui",splittedSignal[0])
    for i in range(int(len(signal)/(interval*200))):
        for j in range(interval*200):
            splittedSignal[i][j]=signal[i*interval*200+j]
    return splittedSignal

def split_hypnogram(hypno,interval = 5):
    hypno_30s = []
    n = (int)(30 / interval)
    for i in range (0,len(hypno)-1,n):
        if(i+n-1>len(hypno)):
            break
        mean = np.mean(np.asarray((hypno[i:i+n])).astype(np.int))
        hypno_30s.append(mean.astype(np.int))
    return hypno_30s

def create_signal_label_arrays(signals,hypno):
    data=[]
    for i in range(len(signals)):
        data.extend([signals[i],hypno[i]])
    return data