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
person = 'katja'
'Definitions'
fs = 6103.515625
groupname = ['NonnoxERP','NoxERP']
i = 0
mean_size=10

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
                        'if you want mean, epoch in range 20, else len(data....)'
                        if np.size(data[groupname[group]]['block'][sets]['channel'][channel]['ERPs']): #If there are stimulations 
    
                            'Get epoch'
                            #stim = data[groupname[group]]['block'][sets]['channel'][channel]['ERPs'][:,epoch]
                            'get mean data'
                            stim=np.mean(data[groupname[group]]['block'][sets]['channel'][channel]['ERPs'][:,j:j+mean_size],axis =1) 
                            stim=(stim-np.mean(stim))/np.std(stim)
                            'Transform to tensor'
                            tempdata = torch.tensor(stim)
                            tempdata = torch.unsqueeze(tempdata,0)
                            if 0 <= sets <=5:
                                if group == 1:
                                    if person == 'hjalte':
                                        os.chdir(r'C:\Users\clark\Desktop\STFT\Val\Nox')
                                    if person == 'mikkel':
                                        os.chdir(r'C:\Users\Mikkel\Desktop\Timeseries\val\Nox')
                                    if person == 'nickolaj':
                                        os.chdir(r'C:\Users\nicko\Desktop\STFT\val\Nox')
                                    if person == 'katja':
                                        os.chdir(r'C:\Users\Katja Stougård\Documents\GitHub\Data\Val\Nox')  
                                    torch.save(tempdata,'val_nox_'+ str(i) +'.pt')
                                else:
                                    if person == 'hjalte':
                                        os.chdir(r'C:\Users\clark\Desktop\STFT\Val\NonNox')
                                    if person == 'mikkel':
                                        os.chdir(r'C:\Users\Mikkel\Desktop\Timeseries\train\Nonnox')
                                    if person == 'nickolaj':
                                        os.chdir(r'C:\Users\nicko\Desktop\STFT\val\Nonnox')
                                    if person == 'katja':
                                        os.chdir(r'C:\Users\Katja Stougård\Documents\GitHub\Data\Val\Nonnox')
                                    torch.save(tempdata,'val_nonnox_'+ str(i) +'.pt')
                            if 9 <= sets <=11:
                                if group == 1:
                                    if person == 'hjalte':
                                        os.chdir(r'C:\Users\clark\Desktop\STFT\Test\Nox')
                                    if person =='mikkel':
                                        os.chdir(r'C:\Users\Mikkel\Desktop\Timeseries\test\Nox')
                                    if person == 'nickolaj':
                                        os.chdir(r'C:\Users\nicko\Desktop\STFT\Test\Nox')
                                    if person == 'katja':
                                        os.chdir(r'C:\Users\Katja Stougård\Documents\GitHub\Data\Test\Nox')
                                    torch.save(tempdata,'test_nox_'+ str(i) +'.pt')
                                else:
                                    if person == 'hjalte':
                                        os.chdir(r'C:\Users\clark\Desktop\STFT\Test\NonNox')
                                    if person == 'mikkel':
                                        os.chdir(r'C:\Users\Mikkel\Desktop\Timeseries\test\Nonnox')
                                    if person == 'nickolaj':
                                            os.chdir(r'C:\Users\nicko\Desktop\STFT\Test\Nonnox')
                                    if person == 'katja':
                                        os.chdir(r'C:\Users\Katja Stougård\Documents\GitHub\Data\Test\Nonnox')

                                    torch.save(tempdata,'test_nonnox_'+ str(i) +'.pt')
                            if 12 <= sets <= 47: 
                                if group == 1:
                                    if person == 'hjalte':
                                        os.chdir(r'C:\Users\clark\Desktop\STFT\Train\Nox')
                                    if person == 'mikkel':
                                        os.chdir(r'C:\Users\Mikkel\Desktop\Timeseries\Train\Nox')
                                    if person == 'nickolaj':
                                        os.chdir(r'C:\Users\nicko\Desktop\STFT\Train\Nox')
                                    if person == 'katja':
                                        os.chdir(r'C:\Users\Katja Stougård\Documents\GitHub\Data\Train\Nox')
                                    
                                    torch.save(tempdata,'train_nox_'+ str(i) +'.pt')
                                else:
                                    if person == 'hjalte':
                                        os.chdir(r'C:\Users\clark\Desktop\STFT\Train\NonNox')
                                    if person == 'mikkel':
                                        os.chdir(r'C:\Users\Mikkel\Desktop\Timeseries\Train\NonNox')
                                    if person == 'nickolaj':
                                        os.chdir(r'C:\Users\nicko\Desktop\STFT\Train\Nonnox')
                                    if person == 'katja':
                                        os.chdir(r'C:\Users\Katja Stougård\Documents\GitHub\Data\Train\Nonnox')
                                    torch.save(tempdata,'train_nonnox_'+ str(i) +'.pt')
                            # stft_dic[groupname[group]]['block'][sets]['channel'][channel]['ERPs'].append(stft_tensor)
                            i +=1 #counter til at gemme filer
                            j = j+mean_size #counter til gennemsnit af STFT
                            'Plot colormap'
                            # if j == 5:
                            #     plt.figure()
                            #     plt.pcolormesh(t, f, Zxx, cmap=cm.plasma)
                            #     plt.xlabel('Time [sec]', fontweight='bold')
                            #     plt.ylabel('Frequency [Hz]', fontweight='bold')    
                            #     plt.colorbar()
                            #     plt.ylim(0,300)
                            
                                 
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
