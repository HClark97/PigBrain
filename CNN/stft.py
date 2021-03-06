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
person = 'hjalte'
'Definitions'
fs = 6103.515625
groupname = ['NonnoxERP','NoxERP']

plot = False
save = True

'STFT definitions'
cutoff = 250
nperseg = 350
w = 'hann'
i = 0
mean_size = 25
zf = 2 #Zoom factor

'Open up data'
for group in range (len(groupname)): #2: nox and nonNox
    for sets in range(len(data['NonnoxERP']['block'])): #48 
        print('Set '+ str(sets+1))
        for channel in range(len(data['NonnoxERP']['block'][0]['channel'])): #32
            if np.size(data['NonnoxERP']['block'][sets]['channel']):
                # stft_dic[groupname[group]]['block'][sets]['channel'][channel]['ERPs']=[]
                #print('Deleted entry in channel '+ str(channel+1))
                j = 0
                for epoch in range(100//mean_size):#len(data['NonnoxERP']['block'][0]['channel'][0]['ERPs'][0])): #100
                        if np.size(data[groupname[group]]['block'][sets]['channel'][channel]['ERPs']): #If there are stimulations 
    
                            'Get epoch'
                            #stim=data[groupname[group]]['block'][sets]['channel'][channel]['ERPs'][:,epoch]
                            'get mean data'
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
                            'Transform to tensor'
                            tempdata = torch.tensor(Zxx)
                            tempdata = torch.unsqueeze(tempdata,0)
                            if 3 <= sets <= 5:
                                if group == 1:
                                    if person == 'hjalte':
                                        os.chdir(r'C:\Users\clark\Desktop\STFT\Val\Nox')
                                    if person == 'mikkel':
                                        os.chdir(r'C:\Users\Mikkel\Desktop\STFT\val\Nox')
                                    if person == 'nickolaj':
                                        os.chdir(r'C:\Users\nicko\Documents\GitHub\STFT\val\Nox')    
                                    if save:
                                        torch.save(tempdata,'val_nox_'+ str(i) +'.pt')
                                else:
                                    if person == 'hjalte':
                                        os.chdir(r'C:\Users\clark\Desktop\STFT\Val\NonNox')
                                    if person == 'mikkel':
                                        os.chdir(r'C:\Users\Mikkel\Desktop\STFT\val\Nonnox')
                                    if person == 'nickolaj':
                                        os.chdir(r'C:\Users\nicko\Documents\GitHub\STFT\val\Nonnox')   
                                    if save:    
                                        torch.save(tempdata,'val_nonnox_'+ str(i) +'.pt')
                            if 9 <= sets <=11:
                                if group == 1:
                                    if person == 'hjalte':
                                        os.chdir(r'C:\Users\clark\Desktop\STFT\Test\Nox')
                                    if person =='mikkel':
                                        os.chdir(r'C:\Users\Mikkel\Desktop\STFT\Test\Nox')
                                    if person == 'nickolaj':
                                        os.chdir(r'C:\Users\nicko\Documents\GitHub\STFT\Test\Nox')
                                    if save:
                                        torch.save(tempdata,'test_nox_'+ str(i) +'.pt')
                                else:
                                    if person == 'hjalte':
                                        os.chdir(r'C:\Users\clark\Desktop\STFT\Test\NonNox')
                                    if person == 'mikkel':
                                        os.chdir(r'C:\Users\Mikkel\Desktop\STFT\Test\Nonnox')
                                    if person == 'nickolaj':
                                            os.chdir(r'C:\Users\nicko\Desktop\STFT\Test\Nonnox')

                                    torch.save(tempdata,'test_nonnox_'+ str(i) +'.pt')
                            if 36 <= sets <= 47 or 12 <= sets <= 14:#3 <= sets <= 5 or 12 <= sets <=20 or 27 <= sets <= 41:  #0 <= sets <= 8 or 
                                if group == 1:
                                    if person == 'hjalte':
                                        os.chdir(r'C:\Users\clark\Desktop\STFT\Train\Nox')
                                    if person == 'mikkel':
                                        os.chdir(r'C:\Users\Mikkel\Desktop\STFT\Train\Nox')
                                    if person == 'nickolaj':
                                        os.chdir(r'C:\Users\nicko\Documents\GitHub\STFT\Train\Nox')
                                    if save:
                                        torch.save(tempdata,'train_nox_'+ str(i) +'.pt')
                                else:
                                    if person == 'hjalte':
                                        os.chdir(r'C:\Users\clark\Desktop\STFT\Train\NonNox')
                                    if person == 'mikkel':
                                        os.chdir(r'C:\Users\Mikkel\Desktop\STFT\Train\NonNox')
                                    if person == 'nickolaj':
                                        os.chdir(r'C:\Users\nicko\Documents\GitHub\STFT\Train\Nonnox')
                                    if save:
                                        torch.save(tempdata,'train_nonnox_'+ str(i) +'.pt')
                            # stft_dic[groupname[group]]['block'][sets]['channel'][channel]['ERPs'].append(stft_tensor)
                            i +=1 #counter til at gemme filer
                            j = j+mean_size #counter til gennemsnit af STFT
            
            'Plot colormap'
            if plot:
                plt.figure()
                plt.title(str(groupname[group])+' Set:'+str(sets+1)+' Channel: '+str(channel+1))
                plt.pcolormesh(t, f, Zxx, cmap=cm.plasma)
                plt.xlabel('Time [sec]', fontweight='bold')
                plt.ylabel('Frequency [Hz]', fontweight='bold')    
                plt.colorbar()

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
