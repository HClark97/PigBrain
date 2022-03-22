
import numpy as np
from scipy import signal
import scipy.fftpack 
from scipy.fft import fft, fftfreq
import matplotlib.pyplot as plt
import FunctionsP7.FFT.Power_density as pdf
import FunctionsP7.filtering.filters as f
import loader as l
from scipy.io import savemat
import Preprocessing.plot as pp

#LOAD DATA
data=l.load_mat_file(1) #<-- til at loade epochs 'nonpreprocessed_data.mat'
fs =  6103.5156
#fs = 6103.5156 # sample frequency
samplesPrEpoch = (fs/1000)*550

noisy_channels = { # First entry is experiment number, second entry is an array of channels to remove
    1: [[18,20,23,24,26,27,28,29,30,31,32],[17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32],[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32]],
    2: [[18,20,23],[18,20,23],[18,20,23]],
    3: [[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32],[],[]],
    4: [[],[],[14]],
    5: [[18,20,23],[18,20,23],[18,20,23]],
    6: [[],[],[]],
    7: [[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32],[],[]],
    8: [[],[],[]],
    9: [[31],[31],[31]],
    10: [[5],[5],[5]],
    11: [[],[],[]],
    12: [[16,18,20,23],[16,18,20,23],[16,18,20,23]],
    13: [[16,18,20,23,29],[18,20,23,29],[18,20,23,29]],
    14: [[9,14,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32],[9,14,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32],[9,14,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32]],
    15: [[18,20,23],[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,18,20,23],[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,18,20,23]],
    16: [[18,20,23,29],[17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32],[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32]]
}

#REMOVAL OF NOISY CHANNELS 
#exp --> set + ((exp-1)*3) --> channel   
for exp in range(1,(round(len(data['NoxERP']['block'])/3)+1)): #16
    for sets in range (3): #3 sets i hver exp 
        if noisy_channels[exp][sets]: #if not empty
              for index in range(len(noisy_channels[exp][sets])): #number of noisy channels fra given exp and set 
                  Nchannel = noisy_channels[exp][sets][index] #channel to remove
                  if np.size(data['NonnoxERP']['block'][sets+((exp-1)*3)]['channel']): #If there are stimulations 
                    data['NonnoxERP']['block'][sets+((exp-1)*3)]['channel'][Nchannel-1]['ERPs'] = [] #removes nonNox set 
                    data['NoxERP']['block'][sets+((exp-1)*3)]['channel'][Nchannel-1]['ERPs'] = [] #removes nox set 

epoch = data['NonnoxERP']['block'][24]['channel'][0]['ERPs'][:,0]

t = np.arange(0,len(epoch),1)/fs #time vector

notch_data = f.filt_filt_nocth_harmonic(epoch,[50,100,150,200], fs)
high_notch_data = f.filt_filt_but(notch_data, 'high', 5, 1, fs) #10 order effective lowpass filter at 400 hz
band_notch_data = f.filt_filt_but(high_notch_data, 'low', 5, 250, fs) #10 order effective lowpass filter at 400 hz


baseTime = round((fs/1000)*50)
baseline = np.mean(band_notch_data[0:baseTime])
baselineCorrected = band_notch_data - baseline

pp.plot_data_fft_log(epoch, t, fs, 'raw epoch', 1, 1)
pp.plot_data_fft_log(band_notch_data, t, fs, 'Filtered epoch', 1, 1)
pp.plot_data_fft_log(baselineCorrected, t, fs, 'Baseline corrected epoch', 1, 1)

# # #PRE-PROCESSING OF ALL EPOCHS (baseline, notch & bandpass)
# groupname = ['NonnoxERP','NoxERP']
# for group in len(groupname): #2: nox and nonNox
#     for sets in range(len(data['NonnoxERP']['block'])): #48 
#         for channel in range(len(data['NonnoxERP']['block'][sets]['channel'])): #32
#             for epoch in range(len(data['NonnoxERP']['block'][sets]['channel'][channel]['ERPs'][0])):
#                 ## Baseline correction fra P7
#                 epoch = data[groupname[group]]['block'][sets]['channel'][channel]['ERPs'][:,epoch] #epoch of 550 ms
#                 # baseline = np.mean(data[group,sets,channel,epoch]) #Taking the first 50 ms of the data for baseline correction for exp, channel and epoch
#                 # # Correct the baseline
#                 # data[group,sets,channel,epoch] = data[exp,channel,epoch] - baseline
                
#                 ## Notch filtering (+harmonics) 
#                 # harmonics = np.array([50,100,150,200,250,300,350])
                
#                 ## Bandpas filtering (1-250Hz) 
#                 # data[group,sets,channel,epoch] = f.filt_filt_nocth_harmonic(data[group,sets,channel,epoch],harmonics, fs)
#                 # data[group,sets,channel,epoch] = f.filt_filt_but(data[group,sets,channel,epoch], 'band', 5, [1,250], fs) #10 order effective lowpass filter at 400 hz

#                 #savemat("preprocessed_data.mat", data)


