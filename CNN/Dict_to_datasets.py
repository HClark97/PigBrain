

import torch as nn
import loader as l
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import scipy
from scipy import signal
from scipy.io import savemat
import copy
import mpu
import plyer as pl



val_data_label = []
val_data =  []
test_data = []
test_data_label = []
train_data = []
train_data_label = []

'Definitions'
fs = 6103.515625
groupname = ['NonnoxERP','NoxERP']
path = pl.filechooser.open_file()
data = mpu.io.read('stft_dict.pickle')
i = 0

for group in range (len(groupname)): #2: nox and nonNox
    for sets in range(len(data['NonnoxERP']['block'])): #48 
        print('Set '+ str(sets+1))
        for channel in range(len(data['NonnoxERP']['block'][0]['channel'])): #32
            if np.size(data['NonnoxERP']['block'][sets]['channel']):
                # stft_dic[groupname[group]]['block'][sets]['channel'][channel]['ERPs']=[]
                #print('Deleted entry in channel '+ str(channel+1))
                for epoch in range(len(data['NonnoxERP']['block'][0]['channel'][0]['ERPs'][0])): #100
                        if np.size(data[groupname[group]]['block'][sets]['channel'][channel]['ERPs']): #If there are stimulations
                              tempdata  = np.asarray(data['NonnoxERP']['block'][sets]['channel'][channel]['ERPs'][0])
                              tempdata = nn.tensor(tempdata)
                              i += 1
                              
                                  
print(i)                                 
# val_dataset = [val_data, val_data_label]  
# test_dataset = [test_data, test_data_label] 
# train_dataset = [train_data, train_data_label]     
# mpu.io.write('valdataset.pickle', val_dataset)
# mpu.io.write('testdataset.pickle', test_dataset)
# mpu.io.write('traindataset.pickle', train_dataset)
