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
fs = 6103.5156 # sample frequency
samplesPrEpoch = (fs/1000)*550

noisy_channels = { # First entry is experiment number, second entry is an array of channels to remove
    1: [[18,20,23,24,26,27,28,29,30,31,32],[17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32],[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32]],
    2: [[18,20,23],[18,20,23],[18,20,23]],
    3: [[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32],[],[]],
    4: [{},{},{14}],
    5: [[18,20,23],[18,20,23],[18,20,23]],
    6: [[],[],[]],
    7: [[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32],[],[]],
    8: [[],[],[]],
    9: [[31],[31],[31]],
    10: [[5],[5],[5]],
    11: [[],[],[]],
    12: [[],[],[]],
    13: [[16,18,20,23],[16,18,20,23],[16,18,20,23]],
    14: [[16,18,20,23,29],[18,20,23,29],[18,20,23,29]],
    15: [[9,14,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32],[9,14,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32],[9,14,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32]],
    16: [[18,20,23],[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,18,20,23],[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,18,20,23]],
    17: [[18,20,23,29],[17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32],[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32]]
}

#REMOVAL OF NOISY CHANNELS 
#groups --> set + ((exp-1)*3) --> channel   
exp = 1
for group in range(len(data)): #2 --> nonNox og Nox
      for sets in range(len(data['NoxERP']['block'])): #48 
          for channel in range(len(data['NonnoxERP']['block'][0]['channel'])): #32 
              for epoch in range(len(data['NonnoxERP']['block'][0]['channel'][0]['ERPs'][0])): #100 stim
                  if noisy_channels[exp][sets] NOT empty:
                      for len noisy_channels[exp][sets]: #number of noisy channels fra given exp and set
                      



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




