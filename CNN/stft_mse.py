# -*- coding: utf-8 -*-
"""
Created on Tue Mar 29 12:40:55 2022

@author: nicko
"""

import torch
import loader as l
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import scipy
from scipy import signal
from scipy.io import savemat
import copy
import mpu
import os
from scipy import ndimage
'Load data'
if not 'data' in globals():
    data=l.load_mat_file()

# stft_dic=copy.deepcopy(data)
val_nonnox_data = []
val_nox_data =  []
train_nonnox_data = []
train_nox_data =  []
test_nonnox_data = []
test_nox_data =  []
person = 'nickolaj'
'Definitions'
fs = 6103.515625
groupname = ['NonnoxERP','NoxERP']

'Plot and Save'
plot = True
save = False

'STFT definitions'
cutoff = 250
nperseg = 350
w = 'hann'
i = 0
mean_size = 100
zf = 2 #Zoom factor
Zxx=np.random.rand(28,42)
setZxx=0
meanmse=0
'Open up data'
for group in range (len(groupname)): #2: nox and nonNox
    for sets in range(len(data['NonnoxERP']['block'])): #48 
        print('Set '+ str(sets+1))
        preZxx=setZxx
        for channel in range(len(data['NonnoxERP']['block'][0]['channel'])): #32
            if np.size(data['NonnoxERP']['block'][sets]['channel']):
                # stft_dic[groupname[group]]['block'][sets]['channel'][channel]['ERPs']=[]
                #print('Deleted entry in channel '+ str(channel+1))
                j = 0
                preZxx=Zxx
                for epoch in range(100//mean_size):#len(data['NonnoxERP']['block'][0]['channel'][0]['ERPs'][0])): #100
                        if np.size(data[groupname[group]]['block'][sets]['channel'][channel]['ERPs']): #If there are stimulations 
    
                            'Get epoch'
                            #stim = data[groupname[group]]['block'][sets]['channel'][channel]['ERPs'][:,epoch]
                            'Get mean data'
                            stim=np.mean(data[groupname[group]]['block'][sets]['channel'][channel]['ERPs'][:,j:j+mean_size-1],axis =1)
                            stim=(stim-np.mean(stim))/np.std(stim) # Z score norm
                            'Calculate STFT'
                            f, t, Zxx = signal.stft(stim, fs, window = w, nperseg = nperseg, noverlap=nperseg//2)
                            cutindex = round(cutoff/(fs/2)*len(Zxx))
                            Zxx = Zxx[0:cutindex,:] # Keep up til 500 Hz
                            f = f[0:cutindex]
                            Zxx = np.abs(Zxx)
                            Zxx = ndimage.zoom(Zxx,zf)
                            f = ndimage.zoom(f,zf)
                            t = ndimage.zoom(t,zf)
                            #Zxx = Zxx/np.max(Zxx) # Normalize 0-1
                            # sets 0-2,6, 21:26 bad
                            j = j+mean_size #counter til gennemsnit af STFT
            
            setZxx=setZxx+Zxx
                            
        setZxx=setZxx/32                        
        mse = np.mean((preZxx-setZxx))**2
        meanmse = meanmse+mse        
        'Plot colormap'
        if plot:
            plt.figure()
            plt.title(str(groupname[group])+' Set: '+str(sets+1)+' Channel: '+str(channel+1) + ' MSE: '+str(mse))
            plt.pcolormesh(t, f, setZxx, cmap=cm.plasma)
            plt.xlabel('Time [sec]', fontweight='bold')
            plt.ylabel('Frequency [Hz]', fontweight='bold')    
            plt.colorbar()
    print(str(group)+':'+str(meanmse/48))
plt.show()
                            
                                 
# torch.save(val_nox_data,'val_nox.pt')
# torch.save(val_nonnox_data,'val_nonnox.pt')
# torch.save(train_nox_data,'train_nox.pt')
# torch.save(train_nonnox_data,'train_nonnox.pt')
# torch.save(test_nox_data,'test_nox.pt')
# torch.save(test_nonnox_data,'test_nonnox.pt')

# mpu.io.write('val_nox.pickle', val_nox_data)
# mpu.io.write('val_nonnox.pickle', val_nonnox_data)
# mpu.io.write('train_nox.pickle', train_nox_data)
# mpu.io.write('train_nonnox.pickle', train_nonnox_data)
# mpu.io.write('test_nox.pickle', test_nox_data)
# mpu.io.write('test_nonnox.pickle', test_nonnox_data)

# unserialized_data = mpu.io.read('filename.pickle')

# mpu.io.write('stft_dict.pickle', stft_dic)
