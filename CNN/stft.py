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
    data=l.load_mat_file()

stft_dic=copy.deepcopy(data)
val_data_label = []
val_data =  []
test_data = []
test_data_label = []
train_data = []
train_data_label = []

'Definitions'
fs = 6103.515625
groupname = ['NonnoxERP','NoxERP']

'STFT definitions'
cutoff = 500
nperseg = 700
w = 'hann'

'Open up data'
for group in range (len(groupname)): #2: nox and nonNox
    for sets in range(len(data['NonnoxERP']['block'])): #48 
        print('Set '+ str(sets+1))
        for channel in range(len(data['NonnoxERP']['block'][0]['channel'])): #32
            if np.size(data['NonnoxERP']['block'][sets]['channel']):
                stft_dic[groupname[group]]['block'][sets]['channel'][channel]['ERPs']=[]
                #print('Deleted entry in channel '+ str(channel+1))
                for epoch in range(len(data['NonnoxERP']['block'][0]['channel'][0]['ERPs'][0])): #100
                        if np.size(data[groupname[group]]['block'][sets]['channel'][channel]['ERPs']): #If there are stimulations 
    
                            'Get epoch'
                            stim=data[groupname[group]]['block'][sets]['channel'][channel]['ERPs'][:,epoch]
                        
                            'Calculate STFT'
                            f, t, Zxx = signal.stft(stim, fs, window = w, nperseg = nperseg, noverlap=nperseg//2)
                            cutindex = round(cutoff/(fs/2)*len(Zxx))
                            Zxx = Zxx[0:cutindex,:] # Keep up til 500 Hz
                            f = f[0:cutindex]
                            Zxx = np.abs(Zxx)
                            Zxx = Zxx/np.max(Zxx)
                            
                            'Transform to tensor'
                            stft_tensor = nn.tensor(Zxx)
                            
                            stft_dic[groupname[group]]['block'][sets]['channel'][channel]['ERPs'].append(stft_tensor)
                                                        
                            'Plot colormap'
                            # plt.figure()
                            # plt.pcolormesh(t, f, Zxx, cmap=cm.plasma)
                            # plt.xlabel('Time [sec]', fontweight='bold')
                            # plt.ylabel('Frequency [Hz]', fontweight='bold')    
                            # plt.colorbar()
                            # plt.ylim(0,300)
                            

for group in range (len(groupname)): #2: nox and nonNox
    for sets in range(len(data['NonnoxERP']['block'])): #48 
        print('Set '+ str(sets+1))
        for channel in range(len(data['NonnoxERP']['block'][0]['channel'])): #32
            if np.size(data['NonnoxERP']['block'][sets]['channel']):
                stft_dic[groupname[group]]['block'][sets]['channel'][channel]['ERPs']=[]
                for epoch in range(len(data['NonnoxERP']['block'][0]['channel'][0]['ERPs'][0])): #100
                        if np.size(data[groupname[group]]['block'][sets]['channel'][channel]['ERPs']): #If there are stimulations 
                              tempdata  = np.asarray(stft_dic['NonnoxERP']['block'][sets]['channel'][channel]['ERPs'][0])
                              if 0 <= sets <=5:
                                  val_data.append(tempdata)
                                  val_data_label.append(group)
                              if 9 <= sets <=11:
                                  test_data.append(tempdata)
                                  test_data_label.append(group)
                              else: 
                                  train_data.append(tempdata)
                                  train_data_label.append(group)
                                  
                                  
val_dataset = [val_data, val_data_label]  
test_dataset = [test_data, test_data_label] 
train_dataset = [train_data, train_data_label]     
mpu.io.write('valdataset.pickle', val_dataset)
mpu.io.write('testdataset.pickle', test_dataset)
mpu.io.write('traindataset.pickle', train_dataset)

# unserialized_data = mpu.io.read('filename.pickle')

# mpu.io.write('stft_dict.pickle', stft_dic)
