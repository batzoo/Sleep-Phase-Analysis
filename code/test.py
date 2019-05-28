from utils import * 
from database import * 

import pyedflib

subject = 1
dim2 = True

# data = pyedflib.EdfReader('..\\..\\Dataset\\PHYSIONET_SLEEPEDFX_TELEMETRY\\subject (1).edf')
data = pyedflib.EdfReader('..\\..\\Dataset\\DREAMS\\subject (1).edf')

print(data.getHeader())
