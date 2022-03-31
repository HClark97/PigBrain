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
if not 'data' in globals():
    data=l.load_mat_file()

'Definitions'
fs = 6103.515625
cutoff = 500
nperseg = 700
w = 'hann'

for i in range(5):
    stim=data['NoxERP']['block'][5]['channel'][0]['ERPs'][:,i]
    
    'Turn stim into tensor'
    # stim_tensor=nn.tensor(stim)
    
    'Calculate STFT'
    # stft=nn.stft(stim_tensor,n_fft,return_complex=True, window=nn.hann_window(n_fft))
    f, t, Zxx = signal.stft(stim, fs, window = w, nperseg = nperseg, noverlap=nperseg//2)
    
    'Cut to 500 Hz'
    cutindex= round(cutoff/(fs/2)*len(Zxx))
    Zxx = Zxx[0:cutindex,:] # Keep up til 500 Hz
    f = f[0:cutindex]
    Zxx = np.abs(Zxx)
    Zxx = Zxx/np.max(Zxx)
    
    'Add absolute value of imaginary numbers to maginary'
    # out_real = stft[:, :, 0]
    # out_imag = stft[:, :, 1]
    # out_abs = nn.sqrt(out_real**2 + out_imag**2)

    'Arrange axis vectors'
    # py_t = np.arange(len(stft[0]))*len(stim)/float(fs)/len(stft[0])
    # py_f = np.arange(0,len(stft))*float(fs/2)/len(stft)
    
    'Plot colormap' 
    plt.figure()
    plt.pcolormesh(t, f, Zxx, cmap=cm.plasma)
    plt.xlabel('Time [sec]', fontweight='bold')
    plt.ylabel('Frequency [Hz]', fontweight='bold')    
    plt.colorbar()