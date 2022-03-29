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

'Load data'
if not 'data' in globals():
    data=l.load_mat_file()

'Definitions'
fs = 6103.515625
n_fft = 256
groupname = ['NonnoxERP','NoxERP']
i=0

'Open up data'
for group in range (len(groupname)): #2: nox and nonNox
    for sets in range(len(data['NonnoxERP']['block'])): #48 
        for channel in range(len(data['NonnoxERP']['block'][0]['channel'])): #32
            for epoch in range(len(data['NonnoxERP']['block'][0]['channel'][0]['ERPs'][0])): #100
                if np.size(data['NonnoxERP']['block'][sets]['channel']):
                    if np.size(data[groupname[group]]['block'][sets]['channel'][channel]['ERPs']): #If there are stimulations 

                        'Get epoch'
                        stim=data[groupname[group]]['block'][sets]['channel'][channel]['ERPs'][:,epoch]
                    
                        'Calculate STFT'
                        f, t, Zxx = signal.stft(stim, fs)
                        Zxx = np.abs(Zxx)
                        Zxx = Zxx/np.max(Zxx)
                        
                        'Transform to tensor'
                        stft_tensor = nn.tensor(Zxx)
                        
                        #data[groupname[group]]['block'][sets]['channel'][channel]['ERPs'][:,epoch] 
                        i += 1
                        data[groupname[group]]['block'][sets]['channel'][channel]['ERPs'][:,:,epoch]= stft_tensor
                        print(i)
                        # 'Plot colormap'
                        # plt.figure()
                        # plt.pcolormesh(t, f, Zxx, cmap=cm.plasma)
                        # plt.xlabel('Time [sec]', fontweight='bold')
                        # plt.ylabel('Frequency [Hz]', fontweight='bold')    
                        # plt.colorbar()
                        # plt.ylim(0,300)

savemat("stft_tensor.mat", data) 
                        
