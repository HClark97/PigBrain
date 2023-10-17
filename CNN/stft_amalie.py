
import torch
import loader as l
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import scipy
from scipy import signal
from scipy.io import savemat
from scipy import ndimage
import copy
import os

'Load data'
if not 'data' in globals():
    data=l.load_mat_file(1)

val_nonnox_data = []
val_nox_data =  []
train_nonnox_data = []
train_nox_data =  []
test_nonnox_data = []
test_nox_data =  []

'Definitions'
fs = 6103.515625
groupname = ['NonnoxERP','NoxERP']
'STFT definitions'
cutoff = 250
nperseg = 350
w = 'hann'
i = 0

mean = 25
intPolFactor = 2.0

'Open up data'
for group in range (len(groupname)): #2: nox and nonNox
    for sets in range(len(data['NonnoxERP']['block'])): #48 
        print('Set '+ str(sets+1))
        for channel in range(len(data['NonnoxERP']['block'][0]['channel'])): #32
            if np.size(data['NonnoxERP']['block'][sets]['channel']):
                # stft_dic[groupname[group]]['block'][sets]['channel'][channel]['ERPs']=[]
                #print('Deleted entry in channel '+ str(channel+1))
                j = 0
                for epoch in range(100//mean): #100
                        if np.size(data[groupname[group]]['block'][sets]['channel'][channel]['ERPs']): #If there are stimulations 
                            'Get epoch'
                            #stim = data[groupname[group]]['block'][sets]['channel'][channel]['ERPs'][:,epoch] #no mean 
                            'get mean data'
                            stim=np.mean(data[groupname[group]]['block'][sets]['channel'][channel]['ERPs'][:,j:j+mean],axis =1) #with mean  
                            #stim=(stim-np.mean(stim))/np.std(stim) # Z score normalisering
                            'Calculate STFT'
                            f, t, Zxx = signal.stft(stim, fs, window = w, nperseg = nperseg, noverlap=nperseg//2)
                            cutindex = round(cutoff/(fs/2)*len(Zxx))
                            Zxx = Zxx[0:cutindex,:] # Keep up til 250 Hz
                            f = f[0:cutindex]
                            Zxx = np.abs(Zxx)
                            Zxx = ndimage.zoom(Zxx,intPolFactor) # interpolation 
                            Zxx = Zxx/np.max(Zxx) #normalize 0-1 
                            #f = ndimage.zoom(f,intPolFactor)
                            #t = ndimage.zoom(t,intPolFactor)
                            
                            'Transform to tensor'
                            tempdata = torch.tensor(Zxx)
                            tempdata = torch.unsqueeze(tempdata,0)
                            if 21 <= sets <=23 or 3 <= sets <=5: #final validation data
                                if group == 1:
                                    os.chdir(r'E:\EXAI data\STFT\Val\Nox')
                                    torch.save(tempdata,'val_nox_'+ str(i) +'.pt')
                                else:
                                    os.chdir(r'E:\EXAI data\STFT\Val\NonNox')
                                    torch.save(tempdata,'val_nonnox_'+ str(i) +'.pt')
                            if 9 <= sets <=11 or 42 <= sets <=44: #final test data
                                if group == 1:
                                    os.chdir(r'E:\EXAI data\STFT\Test\Nox')
                                    torch.save(tempdata,'test_nox_'+ str(i) +'.pt')
                                else:
                                    os.chdir(r'E:\EXAI data\STFT\Test\NonNox')
                                    torch.save(tempdata,'test_nonnox_'+ str(i) +'.pt')
                            if 12 <= sets <= 15 or 19 <= sets <= 20 or 24 <= sets <= 41 or 45 <= sets <= 46: #final train data
                                if group == 1:
                                    os.chdir(r'E:\EXAI data\STFT\Train\Nox')
                                    torch.save(tempdata,'train_nox_'+ str(i) +'.pt')
                                else:
                                    os.chdir(r'E:\EXAI data\STFT\Train\NonNox')
                                    torch.save(tempdata,'train_nonnox_'+ str(i) +'.pt')
                            # stft_dic[groupname[group]]['block'][sets]['channel'][channel]['ERPs'].append(stft_tensor)
                            i +=1 #counter til at gemme filer
                            j = j+mean #counter til gennemsnit af STFT
                            'Plot colormap'
                            # if j == 5:
                            #     plt.figure()
                            #     plt.pcolormesh(t, f, Zxx, cmap=cm.plasma)
                            #     plt.xlabel('Time [sec]', fontweight='bold')
                            #     plt.ylabel('Frequency [Hz]', fontweight='bold')    
                            #     plt.colorbar()
                            #     plt.ylim(0,300)
                            
                                 


