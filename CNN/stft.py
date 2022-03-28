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
import scipy
from scipy import signal

'Load data'
#data=l.load_mat_file()

'Define fs'
fs = 6103.515625
n_fft = 256

for i in range(1):
    stim=data['NoxERP']['block'][5]['channel'][0]['ERPs'][:,i]
    
    'Turn stim into tensor'
    stim_tensor=nn.tensor(stim)
    
    'Calculate STFT'
    stft=nn.stft(stim_tensor,n_fft,return_complex=True, window=nn.hann_window(n_fft))
    f, t, Zxx = signal.stft(stim, fs)
    
    'Add absolute value of imaginary numbers to maginary'
    # out_real = stft[:, :, 0]
    # out_imag = stft[:, :, 1]
    # out_abs = nn.sqrt(out_real**2 + out_imag**2)
    stft=np.abs(stft)

    'Arrange axis vectors'
    py_t = np.arange(len(stft[0]))*len(stim)/float(fs)/len(stft[0])
    py_f = np.arange(0,len(stft))*float(fs/2)/len(stft)
    
    'Plot colormap'
    plt.figure()
    plt.pcolormesh(py_t,py_f,stft, cmap=cm.plasma)
    plt.xlabel('Time [sec]', fontweight='bold')
    plt.ylabel('Frequency [Hz]', fontweight='bold')
    plt.colorbar()
    plt.ylim(0,500)
    
    
    plt.figure()
    plt.pcolormesh(t,f,np.abs(Zxx), cmap=cm.plasma)
    plt.xlabel('Time [sec]', fontweight='bold')
    plt.ylabel('Frequency [Hz]', fontweight='bold')    
    plt.colorbar()
    plt.ylim(0,500)
