# -*- coding: utf-8 -*-
"""
Created on Mon Apr  4 10:30:23 2022

@author: mbj
"""
import numpy as np
import matplotlib.pyplot as plt
import loader as l
from matplotlib.ticker import FormatStrFormatter
'Load data'
if not 'data' in globals():
    data=l.load_mat_file(1)

fs = np.float64(6103.515625)

t = (np.arange(0,len(data['NonnoxERP']['block'][0]['channel'][0]['ERPs']),1)/fs)-0.05
groupname = ['NonnoxERP','NoxERP']
for group in range (1):#len(groupname)): #2: nox and nonNox
    for sets in range(45,46):#len(data['NonnoxERP']['block'])): #48 
        for channel in range(15,16):#len(data['NonnoxERP']['block'][0]['channel'])): #32
            if np.size(data['NonnoxERP']['block'][sets]['channel']):
                if np.size(data[groupname[group]]['block'][sets]['channel'][channel]['ERPs']): #If there are stimulations
                    meandata = np.mean(data[groupname[1]]['block'][sets]['channel'][channel]['ERPs'],axis = 1)
                    meandata = meandata*1000 #To get values in microvolt
                    fig,ax = plt.subplots()
                    plt.plot(t,meandata)
                    plt.ylabel('Amplitude [µV]',fontweight = 'bold')
                    plt.xlabel('Time [s]',fontweight = 'bold')
                    plt.xlim(-0.05,0.5)
                    plt.ylim(-0.02,0.04)
                    ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
                    #plt.title('Stim: '+ str(groupname[group]) +' Set: ' + str(sets+1) + ' Channel: ' + str(channel+1))
                    #fig.savefig('{}.pdf'.format('preprocessed_mean_epoch' + '_' + str(sets) + '_' + str(channel)),format='pdf')