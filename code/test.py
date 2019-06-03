from utils import * 
from database import * 
import mne.io.edf

subject = 1
dim2 = True

mescouilles, data, mescouilles = mne.io.edf.read_raw_edf("..\\..\\Dataset\\DREAMS\\subject (1).edf")
# data = mne.io.edf.read_edf_header("..\\..\\Dataset\\DREAMS\\subject (1).edf")
# data = pyedflib.EdfReader("..\\..\\Dataset\\DREAMS\\subject (1).edf")
print(data)