import tdt
import numpy as np
from scipy import signal
import scipy.fftpack 
from scipy.fft import fft, fftfreq
import matplotlib.pyplot as plt
import FunctionsP7.FFT.Power_density as pdf
import FunctionsP7.filtering.filters as butfilt
import loader as l
import mergeFunction as mf

#LOAD DATA
# data1=l.load_mat_file(1) #<-- til at loade epochs fra subject 1
# data2=l.load_mat_file(2) #<-- til at loade epochs fra subject 9
# data3=l.load_mat_file(3) #<-- til at loade epochs fra subject 1 without RS4

data=l.load_mat_file(1) #<-- loader alt data 
fs = 6103.5156 # sample frequency

#REMOVAL OF NOISY CHANNELS 
# groups --> set + ((exp-1)*3) --> channel   
#             
# for group in range(np.size([Non,Nox])):
#      for sets in range(np.shape(data['NoxERP']['block'])):
#          for channel in range(np.shape(data['NonnoxERP']['block'][0]['channel'])):
#              for epoch in range(np.size(Nox['block'][0]['channel'][0]['ERPs'])):


# #PRE-PROCESSING OF ALL EPOCHS (baseline, notch & bandpass)
# for group in range(np.size([Non,Nox])):
#     for sets in range(np.shape(Nox['block'])):
#         for channel in range(np.shape(data['NonnoxERP']['block'][0]['channel'])):
#             for epoch in range(np.size(Nox['block'][0]['channel'][0]['ERPs'])):
#                 ## Baseline correction fra P7
#                 # baseline = np.mean(data[group,sets,channel,epoch]) #Taking the first 50 ms of the data for baseline correction for exp, channel and epoch
#                 # # Correct the baseline
#                 # data[group,sets,channel,epoch] = data[exp,channel,epoch] - baseline
                
#                 ## Notch filtering (+harmonics) 
#                 # harmonics = np.array([50,100,150,200,250,300,350])
                
#                 ## Bandpas filtering (1-250Hz) 
#                 # data[group,sets,channel,epoch] = f.filt_filt_nocth_harmonic(data[group,sets,channel,epoch],harmonics, fs)
#                 # data[group,sets,channel,epoch] = f.filt_filt_but(data[group,sets,channel,epoch], 'band', 5, [1,250], fs) #10 order effective lowpass filter at 400 hz




