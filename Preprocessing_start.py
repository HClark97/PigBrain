## USE THIS FILE TO PRE-PROCESSES EPOCHS 

import numpy as np
import FunctionsP7.filtering.filters as f
import loader as l

#LOAD DATA
data=l.load_mat_file(1) #<-- til at loade epochs 'nonpreprocessed_data.mat'
fs = np.float64(6103.515625) #sample frequency, fs =  6103.5156
#samplesPrEpoch = (fs/1000)*550

filtered_data = np.zeros(3358, dtype=np.float64)
temp_epoch = np.zeros(3358, dtype=np.float64)

#NOISY CHANNELS IDENTIFIED 
noisy_channels = { # First entry is experiment number, second entry is 3 arrays of channels to remove for set 1, 2 and 3 respectively
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
for exp in range(1,(round(len(data['NoxERP']['block'])/3)+1)): #16 exp
    for sets in range (3): #3 sets i hver exp 
        if noisy_channels[exp][sets]: #if not empty
              for index in range(len(noisy_channels[exp][sets])): #number of noisy channels fra given exp and set 
                  Nchannel = noisy_channels[exp][sets][index] #channel to remove
                  if np.size(data['NonnoxERP']['block'][sets+((exp-1)*3)]['channel']): #If there are stimulations 
                    data['NonnoxERP']['block'][sets+((exp-1)*3)]['channel'][Nchannel-1]['ERPs'] = [] #removes nonNox set 
                    data['NoxERP']['block'][sets+((exp-1)*3)]['channel'][Nchannel-1]['ERPs'] = [] #removes nox set 


# FILTERING OF ALL EPOCHS (notch 50 Hz, notch harmonics & bandpass 1-500)
groupname = ['NonnoxERP','NoxERP']
for group in range (len(groupname)): #2: nox and nonNox
    for sets in range(len(data['NonnoxERP']['block'])): #48 
        for channel in range(len(data['NonnoxERP']['block'][0]['channel'])): #32
            for epoch in range(len(data['NonnoxERP']['block'][0]['channel'][0]['ERPs'][0])): #100
                if np.size(data['NonnoxERP']['block'][sets]['channel']):
                    if np.size(data[groupname[group]]['block'][sets]['channel'][channel]['ERPs']): #If there are stimulations 
                        temp_epoch = data[groupname[group]]['block'][sets]['channel'][channel]['ERPs'][:,epoch] #epoch of 550 ms
                        filtered_data = f.filt_filt_nocth_harmonic(temp_epoch,[50,100,150,200,250,300,350,400,450], fs) #garmonics filtering 
                        filtered_data = f.filt_filt_but(filtered_data, 'stop', 8, [48,52], fs) #16 order effective bandstop filter around 50 Hz
                        filtered_data = f.filt_filt_but(filtered_data, 'high', 5, 1, fs) #10 order effective highpass filter at 1 Hz
                        filtered_data = f.filt_filt_but(filtered_data, 'low', 2, 500, fs) #10 order effective lowpass filter at 250 hz
                        data[groupname[group]]['block'][sets]['channel'][channel]['ERPs'][:,epoch] = filtered_data
                    
savemat("preprocessed_data.mat", data) 

    

