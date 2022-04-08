# -*- coding: utf-8 -*-
"""
Created on Tue Mar 29 12:40:55 2022

@author: nicko
"""

import torch as nn
import loader as l
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import scipy
from scipy import signal
from scipy.io import savemat
import copy
import mpu


'Load data'
if not 'data' in globals():
    data=l.load_mat_file(1)

# stft_dic=copy.deepcopy(data)
val_nonnox_data = []
val_nox_data =  []
train_nonnox_data = []
train_nox_data =  []
test_nonnox_data = []
test_nox_data =  []

'Definitions'
fs = 6103.515625
groupname = ['NonnoxERP','NoxERP']

'STFT definitions'
cutoff = 500
nperseg = 350
w = 'hann'
i = 0
'Open up data'
for group in range (len(groupname)): #2: nox and nonNox
    for sets in range(len(data['NonnoxERP']['block'])): #48 
        print('Set '+ str(sets+1))
        for channel in range(len(data['NonnoxERP']['block'][0]['channel'])): #32
            if np.size(data['NonnoxERP']['block'][sets]['channel']):
                # stft_dic[groupname[group]]['block'][sets]['channel'][channel]['ERPs']=[]
                #print('Deleted entry in channel '+ str(channel+1))
                for epoch in range(len(data['NonnoxERP']['block'][0]['channel'][0]['ERPs'][0])): #100
                        if np.size(data[groupname[group]]['block'][sets]['channel'][channel]['ERPs']): #If there are stimulations 
    
                            'Get epoch'
                            stim=data[groupname[group]]['block'][sets]['channel'][channel]['ERPs'][:,epoch]
                            i += 1
                            'Calculate STFT'
                            f, t, Zxx = signal.stft(stim, fs, window = w, nperseg = nperseg, noverlap=nperseg//2)
                            cutindex = round(cutoff/(fs/2)*len(Zxx))
                            Zxx = Zxx[0:cutindex,:] # Keep up til 500 Hz
                            f = f[0:cutindex]
                            Zxx = np.abs(Zxx)
                            Zxx = Zxx/np.max(Zxx)
                            
                            'Transform to tensor'
                            tempdata = nn.tensor(Zxx)
                            if 0 <= sets <=5:
                                if group == 1:
                                    val_nox_data.append(tempdata)
                                else:
                                    val_nonnox_data.append(tempdata)
                            if 9 <= sets <=11:
                                if group == 1:
                                    test_nox_data.append(tempdata)
                                else:
                                    test_nonnox_data.append(tempdata)
                            if 12 <= sets <= 47: 
                                if group == 1:
                                    train_nox_data.append(tempdata)
                                else:
                                    train_nonnox_data.append(tempdata)
                            # stft_dic[groupname[group]]['block'][sets]['channel'][channel]['ERPs'].append(stft_tensor)
                                                        
                            'Plot colormap'
                            # plt.figure()
                            # plt.pcolormesh(t, f, Zxx, cmap=cm.plasma)
                            # plt.xlabel('Time [sec]', fontweight='bold')
                            # plt.ylabel('Frequency [Hz]', fontweight='bold')    
                            # plt.colorbar()
                            # plt.ylim(0,300)
                            
print(i)
                                 
   
mpu.io.write('val_nox.pickle', val_nox_data)
mpu.io.write('val_nonnox.pickle', val_nonnox_data)
mpu.io.write('train_nox.pickle', train_nox_data)
mpu.io.write('train_nonnox.pickle', train_nonnox_data)
mpu.io.write('test_nox.pickle', test_nox_data)
mpu.io.write('test_nonnox.pickle', test_nonnox_data)

# unserialized_data = mpu.io.read('filename.pickle')

# mpu.io.write('stft_dict.pickle', stft_dic)
