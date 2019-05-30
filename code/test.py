from utils import * 
from database import * 

import mne

subject = 1
dim2 = True

data = np.load("..\\..\\numpy_files\\DREAMS\\CZ2-A1frequency101dimsubject1.npy")
print(np.shape(data))
print(data[0][0])