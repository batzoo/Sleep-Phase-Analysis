import utils
import time
import pyedflib

from utils import * 
from database import *


temp = time.time()
# displaySpectrum_bis(data[50][0],375)


def compute_PSG_data(pathPSG,ind_signaux,frequency):
	filePSG = pyedflib.EdfReader(pathPSG)
	data = []
	for i in range(len(ind_signaux)):
		temp = filePSG.readSignal(i)
		if(frequency!=100):
			temp = resample(frequency,100,temp)
		data.append(splitSignal_notime(30,temp,100))
	return data


data = compute_PSG_data("..\\..\\Dataset\\DREAMS\\subject (1).edf",[1,2,15],200)


print("temps : ",time.time()-temp," secondes")