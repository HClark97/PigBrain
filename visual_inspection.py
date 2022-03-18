# -*- coding: utf-8 -*-

import tdt
import numpy as np
from scipy import signal
import scipy.fftpack 
from scipy.fft import fft, fftfreq
import matplotlib.pyplot as plt
import FunctionsP7.FFT.Power_density as pdf
import FunctionsP7.filtering.filters as f
import FunctionsP7.filtering.plot_data as pd
from numpy.fft import rfft, rfftfreq
import Preprocessing.plot as p

fontsize_title = 25
fontsize_labels = 15

#load data
#data = tdt.read_block(r'C:\Users\mbj\Desktop\Uni\8. semester\Projekt\Data\Subject9-210426-124955')
data = tdt.read_block(r'C:\Users\Mikkel\Desktop\Student data\Without RS4\Subject1-211213-102734')
#data = tdt.read_block(r'/Users/amaliekoch/Desktop/Subject1-210914-103420')

exp = 17
channel = 1

fs = data.streams.Wav1.fs#RSn1.fs # sample frequency 
data = data.streams.Wav1.data#RSn1.data #data 
t = np.arange(0,len(data[0,:]),1)/fs #time vector

# Harmonic notch filtering
harmonics = np.array([50,100,150,200,250,300,350])
#notch_data = np.zeros(np.shape(data))
#notch_data = f.filt_filt_nocth_harmonic(data,harmonics, fs)

# HIGHPASS AND LOW FILTERING 
#high_notch_data = f.filt_filt_but(notch_data, 'high', 5, 1, fs) #10 order effective highpass filter at 1 hz
#low_high_notch_data = f.filt_filt_but(high_notch_data, 'low', 5, 400, fs) #10 order effective lowpass filter at 400 hz
#band_notch_data = f.filt_filt_but(notch_data, 'band', 5, [1,250], fs) #10 order effective lowpass filter at 400 hz


for i in range (len(data)):
    notch_data = f.filt_filt_nocth_harmonic(data[i],harmonics, fs)
    band_notch_data = f.filt_filt_but(notch_data, 'band', 5, [1,250], fs) #10 order effective lowpass filter at 400 hz
    p.plot_data_fft_log(band_notch_data,
                        t,
                        fs,
                        'Bandpass and notch filtered data in time and frequency domain. ',
                        i,
                        exp)



