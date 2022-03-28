# -*- coding: utf-8 -*-
"""
Created on Tue Mar 22 11:26:24 2022

@author: nicko
"""

import torch as nn
import loader as l
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np


'Load data'
#data=l.load_mat_file()

for i in range(100):
    stim=data['NoxERP']['block'][5]['channel'][0]['ERPs'][:,i]
    
    'Turn stim into tensor'
    stim_tensor=nn.tensor(stim)
    
    'Calculate STFT'
    stft=nn.stft(stim_tensor,1500)
    
    'Add absolute value of imaginary numbers to maginary'
    out_real = stft[:, :, 0]
    out_imag = stft[:, :, 1]
    out_abs = nn.sqrt(out_real**2 + out_imag**2)
    
    'Define fs and arrange time vector'
    fs = 6103.515625
    t = np.arange(len(out_abs[0])+1) / float(fs)
    f = np.arange(0,len(out_abs)+1)*float(fs)/len(out_abs)
    
    
    'Plot colormap'
    plt.figure()
    plt.pcolormesh(out_abs, cmap=cm.plasma)
    plt.xlabel('Time [sec]', fontweight='bold')
    plt.ylabel('Frequency [Hz]', fontweight='bold')
    ylim(0,2)
    #plt.xticks(t)
    #plt.yticks(f)
    plt.colorbar()
