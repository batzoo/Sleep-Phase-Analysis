import numpy as np
from utils import *

name_signaux_interessants = ['FP1-A2','CZ-A1','O1-A2','FP2-A1','O2-A1','CZ2-A1','EMG1','EMG2','EMG3','EOG1','EOG2']

subject = 1 
signal_index = 1 #indice signal voulu
database_folder = '..\\..\\Pologne\\Dataset'

pathEDF = database_folder+'subject'+str(subject)+'.edf'
pathHypnogram = database_folder+'HypnogramAASM_subject'+str(subject)+'.txt'

raw_data_EDF = edfDataExtraction_interestingSignals(pathEDF)
raw_hypnogram = hypnogramDataExtraction(pathHypnogram)

hypnogram = split_hypnogram(raw_hypnogram,5)

signal = raw_data_EDF[signal_index]

data = create_signal_label_arrays(signal,200,hypnogram)
displaySignal_bis(data[600][0],200)

signalDisplay = splitSignal(30,signal)
displaySignal(signalDisplay[600])

displaySignal(signalSmoother(signalDisplay(600), 30))

# spectrum = spectrumCalculation(signalDisplay)
# displaySpectrum(spectrum)