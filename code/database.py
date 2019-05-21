import numpy as np
from utils import *
from database_extraction import *
import pyedflib

name_signaux_interessants = ['FP1-A2','CZ-A1','O1-A2','FP2-A1','O2-A1','CZ2-A1','EMG1','EMG2','EMG3','EOG1','EOG2']

subject = 1 
signal_index = 1 #indice signal voulu
database_folder = '..\\..\\Dataset\\'

pathEDF = database_folder+'subject'+str(subject)+'.edf'

fileEDF = pyedflib.EdfReader(pathEDF) #Openning the EDF file
print(fileEDF.getPatientName())

fileEDF._close() #Closing the EDF file