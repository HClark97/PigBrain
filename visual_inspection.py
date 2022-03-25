## THIS FILE IS USED FOR VISUAL INSPECTION OF THE DATA 
# Aim: to identify noisy channels

import tdt
import numpy as np
from scipy import signal
import FunctionsP7.filtering.filters as f
import FunctionsP7.filtering.plot_data as pd
from numpy.fft import rfft, rfftfreq
import Preprocessing.plot as p

fontsize_title = 25
fontsize_labels = 15

#LOAD DATA (full experiment)
#data = tdt.read_block(r'C:\Users\mbj\Desktop\Uni\8. semester\Projekt\Data\Subject9-210426-124955')
data = tdt.read_block(r'C:\Users\mbj\Desktop\Uni\8. semester\Projekt\Data\Subject9-210427-112449')
#data = tdt.read_block(r'/Users/amaliekoch/Desktop/Subject1-210914-103420')

exp = 17 

## Data with RS4
fs = data.streams.RSn1.fs # sample frequency 
data = data.streams.RSn1.data #data 
t = np.arange(0,len(data[0,:]),1)/fs #time vector

## Data without RS4
#fs = data.streams.Wav1.fs#RSn1.fs # sample frequency 
#data = data.streams.Wav1.data#RSn1.data #data 
#t = np.arange(0,len(data[0,:]),1)/fs #time vector

## Harmonics til filtrering
harmonics = np.array([50,100,150,200,250,300,350,400,450])

for i in range (len(data)): #32 kanaler
    notch_data = f.filt_filt_nocth_harmonic(data[i],harmonics, fs) #Notch filter at 50 Hz + 8 harmonics
    band_notch_data = f.filt_filt_but(notch_data, 'band', 5, [1,250], fs) #10 order effective bandpass filter at 1-250 hz
    
    #Plot data in time and frequency domain
    p.plot_data_fft_log(band_notch_data,
                        t,
                        fs,
                        'Bandpass and notch filtered data in time and frequency domain. ',
                        i,
                        exp)



