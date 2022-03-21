import tdt
import numpy as np
from scipy import signal
import scipy.fftpack 
from scipy.fft import fft, fftfreq
import matplotlib.pyplot as plt
import FunctionsP7.FFT.Power_density as pdf
import FunctionsP7.filtering.filters as butfilt
import loader as l
from scipy.io import savemat

#LOAD DATA
data=l.load_mat_file(1) #<-- til at loade epochs 'nonpreprocessed_data.mat'

noisy_channels = { # First entry is experiment number, second entry is an array of channels to remove
    1: [{18,20,23,24,np.arange(26,33,1)},{np.arange(17,33,1)},{np.arange(1,33,1)}],
    2: [{18,20,23},{18,20,23},{18,20,23}],
    3: [{np.arange(1,33,1)},{},{}],
    4: [{},{},{14}],
    5: [{18,20,23},{18,20,23},{18,20,23}],
    6: [{},{},{}],
    7: [{{np.arange(1,33,1)}},{},{}],
    8: [{},{},{}],
    9: [{31},{31},{31}],
    10: [{5},{5},{5}],
    11: [{},{},{}],
    12: [{},{},{}],
    13: [{16,18,20,23},{16,18,20,23},{16,18,20,23}],
    14: [{16,18,20,23,29},{18,20,23,29},{18,20,23,29}],
    15: [{9,14,np.arange(16,33,1)},{9,14,np.arange(16,33,1)},{9,14,np.arange(16,33,1)}],
    16: [{18,20,23},{np.arange(1,17,1),18,20,23},{np.arange(1,17,1),18,20,23}],
    17: [{18,20,23,29},{np.arange(17,33,1)},{np.arange(1,33,1)}],
}

#REMOVAL OF NOISY CHANNELS 
# groups --> set + ((exp-1)*3) --> channel   
# for group in range(np.size(data)):
#       for sets in range(np.shape(data['NoxERP']['block'])):
#           for channel in range(np.shape(data['NonnoxERP']['block'][0]['channel'])):
#               for epoch in range(np.size(Nox['block'][0]['channel'][0]['ERPs'])):


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




