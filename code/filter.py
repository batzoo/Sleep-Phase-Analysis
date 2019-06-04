######################## Modules a importer ########################

from utils import *
from database import *
import matplotlib.pyplot as plt
import numpy as np
import mne

######################## Definition de la fonction ########################

def frequency_filter(filter, signal, thirty_sec_period_index):

    data = load_data(signal)

    if(thirty_sec_period_index>(len(data)-1)):
        print('Error - Parameter out of range')
        return(-1)

    else:

        filtered_data = []
        filtered_data_1 = []
        filtered_data_2 = []

        values=data[thirty_sec_period_index][0]
        label=data[thirty_sec_period_index][1]

        if (filter == 'delta'):

            for i in range(len(values)):
                if(i>4*(375/50)):
                    filtered_data_1.append(0)
                else:
                    filtered_data_1.append(values[i])

        if (filter == 'theta'):

            for i in range(len(values)):
                if (i<4*(375/50)) or (i>7*(375/50)):
                    filtered_data_1.append(0)
                else:
                    filtered_data_1.append(values[i])

        if (filter == 'alpha'):

            for i in range(len(values)):
                if (i<8*(375/50)) or (i>13*(375/50)):
                    filtered_data_1.append(0)
                else:
                    filtered_data_1.append(values[i])

        if (filter == 'beta'):

            for i in range(len(values)):
                if(i<13*(375/50)):
                    filtered_data_1.append(0)
                else:
                    filtered_data_1.append(values[i])



        for i in range(len(values)):
            filtered_data_2.append(i*(50/375))

        filtered_data.append(filtered_data_1)
        filtered_data.append(filtered_data_2)
        filtered_data.append(label)
        print(filtered_data)
        return(filtered_data)


######################## Exemples d'utilisation de la fonction ########################

displaySpectrum(frequency_filter('delta','CZ2-A1frequency101dimsubject1',550))
displaySpectrum(frequency_filter('theta','CZ2-A1frequency101dimsubject1',22))
displaySpectrum(frequency_filter('alpha','CZ2-A1frequency101dimsubject1',47))
displaySpectrum(frequency_filter('beta','CZ2-A1frequency101dimsubject1',3))

# La fonction prend en paramètres:
# - n°1 = Le type de filtrage (delta,theta,alpha,beta)
# - n°2 = Le nom du fichier .npy dont le signal correspondant est a filtrer
# - n°3 = L indice de la periode de 30 secondes a analyser dans le fichier

# La fonction retourne:
# - Argument[0] = la liste des valeurs Level (en y) filtrees (donc avec plein de zero)
# - Argument[1] = la liste des valeurs Frequence (en x) qui vont de 0 a 50 Hz
# - Argument[3] = le label de sleep_stage correspondant a la periode de 30 secondes analysee