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
from scipy import ndimage
'Load data'
if not 'data' in globals():
    data=l.load_mat_file()

'Definitions'
fs = 6103.515625
cutoff = 250
nperseg = 350
w = 'hann'

for i in range(1):
    #stim=data['NoxERP']['block'][45]['channel'][15]['ERPs'][:,0]
    
    stim = np.mean(data['NoxERP']['block'][40]['channel'][15]['ERPs'],axis = 1)
    'Turn stim into tensor'
    # stim_tensor=nn.tensor(stim)
    
    'Calculate STFT'
    # stft=nn.stft(stim_tensor,n_fft,return_complex=True, window=nn.hann_window(n_fft))
    f, t, Zxx = signal.stft(stim, fs, window = w, nperseg = nperseg, noverlap=nperseg//2)
    
    'Cut to 500 Hz'
    cutindex= round(cutoff/(fs/2)*len(Zxx))
    Zxx = Zxx[0:cutindex,:] # Keep up til 500 Hz
    f = f[0:cutindex]
    f = ndimage.zoom(f,10.0)
    t = ndimage.zoom(t,10.0)
    Zxx = np.abs(Zxx)
    Zxx = ndimage.zoom(Zxx,10.0)
    Zxx = Zxx/np.max(Zxx)
    
    'Add absolute value of imaginary numbers to maginary'
    # out_real = stft[:, :, 0]
    # out_imag = stft[:, :, 1]
    # out_abs = nn.sqrt(out_real**2 + out_imag**2)

    'Arrange axis vectors'
    # py_t = np.arange(len(stft[0]))*len(stim)/float(fs)/len(stft[0])
    # py_f = np.arange(0,len(stft))*float(fs/2)/len(stft)
    
    'Plot colormap' 
    fig = plt.figure()
    #plt.figure()
    #plt.axvline(x=0.05,color = 'green')
    plt.pcolormesh((t-0.05), f, Zxx, cmap=cm.plasma) #Vi ændre t, så stimulationen sker i 0.0
    plt.xlabel('Time [sec]', fontweight='bold')
    plt.ylabel('Frequency [Hz]', fontweight='bold')    
    plt.colorbar()
    #fig.savefig('{}.eps'.format('nox_stft'),format='eps')