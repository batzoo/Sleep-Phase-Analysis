from utils import * 
from database import * 
import mne.io.edf

subject = 1
dim2 = True

data = mne.io.edf.read_raw_edf_perso("..\\..\\Dataset\\DREAMS\\subject (1).edf")
# data = mne.io.edf.read_edf_header("..\\..\\Dataset\\DREAMS\\subject (1).edf")
# data = pyedflib.EdfReader("..\\..\\Dataset\\DREAMS\\subject (1).edf")
data = mne.io.edf.read_raw_edf_perso("..\\..\\Dataset\\PHYSIONET_SLEEPEDFX_TELEMETRY\\subject (1).edf")

def get_subject_info(PSGpath):
	gender = ''
	age = 0
	if(utils.DATABASE == 'PHYSIONET_SLEEPEDFX_TELEMETRY'):
		subject_info = mne.io.edf.read_raw_edf_perso(PSGpath)
		gender = subject_info[2][2:subject_info[2].find('_')]
		age = subject_info[2][subject_info[2].find('_')+1:subject_info[2].find('y')]
		if(gender == 'Male'):
			gender = 0
		elif(gender == 'Female'):
			gender = 1
	elif(utils.DATABASE == 'DREAMS'):
		subject_info = mne.io.edf.read_raw_edf_perso(PSGpath)
		gender = subject_info[2][:subject_info[2].find('-')-1] 
		birth = subject_info[2][subject_info[2].find('-')+8:subject_info[2].find('/')+8]
		age = 2002 - (int)(birth)
		if(gender == 'man'):
			gender = 0
		elif(gender == 'woman'):
			gender = 1
		else :
			raise(Exception) 
	return gender, age