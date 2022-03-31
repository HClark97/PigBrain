# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import loader as l
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import scipy
from scipy import signal
from scipy.io import savemat 
import copy

'Load data'
if not 'data' in globals():
    data=l.load_mat_file()
fs = np.float64(6103.515625)
t = np.arange(0,len(data['NonnoxERP']['block'][0]['channel'][0]['ERPs']),1)/fs

groupname = ['NonnoxERP','NoxERP']
for group in range (len(groupname)): #2: nox and nonNox
    for sets in range(len(data['NonnoxERP']['block'])): #48 
        for channel in range(len(data['NonnoxERP']['block'][0]['channel'])): #32
            if np.size(data['NonnoxERP']['block'][sets]['channel']):
                if np.size(data[groupname[group]]['block'][sets]['channel'][channel]['ERPs']):
                    mean_data = np.mean(data[groupname[group]]['block'][sets]['channel'][channel]['ERPs'],axis = 1)
                    plt.figure()
                    plt.plot(t,mean_data)
                    plt.title('Stim: ' + str(groupname[group]) + ' Set: '+ str(sets+1)+' Channel: ' + str(channel+1))
                    plt.xlabel('Time [s]')
                    plt.ylabel('Amplitude [mV]')