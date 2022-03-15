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
#data = tdt.read_block(r'C:\Users\mbj\Desktop\Uni\8. semester\Projekt\Data\Subject9-210526-093759')
data = tdt.read_block(r'/Users/amaliekoch/Desktop/Subject1-210914-103420')

exp = 1
channel = 1

fs = data.streams.RSn1.fs # sample frequency 
data = data.streams.RSn1.data #data 
t = np.arange(0,len(data[0,:]),1)/fs #time vector

#PLOT RAW DATA 
#p.plot_data_fft_log(data, t, fs, 'Raw data in time and frequency domain. ', channel, exp)

# Harmonic notch filtering
harmonics = np.array([50,100,150,200,250,300,350])
notch_data = np.zeros(np.shape(data))
notch_data = f.filt_filt_nocth_harmonic(data,harmonics, fs)

# HIGHPASS AND LOW FILTERING 
#high_notch_data = f.filt_filt_but(notch_data, 'high', 5, 1, fs) #10 order effective highpass filter at 1 hz
#low_high_notch_data = f.filt_filt_but(high_notch_data, 'low', 5, 400, fs) #10 order effective lowpass filter at 400 hz
band_notch_data = f.filt_filt_but(notch_data, 'band', 5, [1,400], fs) #10 order effective lowpass filter at 400 hz


for i in range (len(data)):
    p.plot_data_fft_log(band_notch_data, t, fs, 'Bandpass and notch filtered data in time and frequency domain. ', i, exp)


# for i in range (8):
#     #PLOT AF RAW DATA I TID OG FREKVENS: 
#     fig = plt.figure(figsize=(25, 20))
#     plt.suptitle(('Raw data in time and frequency domain. ' + 'Channel: '+ str(i+1) + ' Exp ' + str(exp)),fontsize =25,fontweight='bold')
#     plt.rc('xtick', labelsize=fontsize_labels)
#     plt.rc('ytick', labelsize=fontsize_labels)
#     plt.subplot(311)
#     plt.plot(t,low_notch_data[i,:])
#     #plt.xlim([0,2])  #only first part of data 
#     #plt.ylim([-500,500])
#     plt.grid(True)
#     plt.title("Time domain",fontsize = fontsize_labels,fontweight='bold')
#     plt.ylabel('Amplitude [µV]',fontsize = fontsize_labels,fontweight='bold')
#     plt.xlabel('Time [s]',fontsize = fontsize_labels,fontweight='bold')
#     plt.subplot(312)
#     raw_data_fft = abs(rfft(low_notch_data[0]))
#     freqs = rfftfreq(low_notch_data[0].size, 1 / fs)
#     plt.plot(freqs, abs(raw_data_fft))
#     plt.title("Frequency domain (FFT)",fontsize = fontsize_labels,fontweight='bold')
#     plt.xlabel('Frequency [Hz]',fontsize = fontsize_labels,fontweight='bold')
#     plt.ylabel('Amplitude',fontsize = fontsize_labels,fontweight='bold')
#     #plt.ylim(-5,300)
#     plt.xlim([-5,300])
#     plt.subplot(313)
#     plt.semilogy(freqs, abs(raw_data_fft**2))
#     plt.grid(True)
#     plt.grid(b = True, which = 'minor', color = 'k', linestyle = '-', alpha = 0.2)
#     plt.minorticks_on()
#     plt.title("Logarithmic frequency domain",fontsize = fontsize_labels,fontweight='bold')
#     plt.ylabel('Power [µV^2]',fontsize = fontsize_labels,fontweight='bold')
#     plt.xlabel('Frequency [Hz]',fontsize = fontsize_labels,fontweight='bold')
#     #plt.ylim(-5,300)
#     plt.xlim([-2,300])
#     plt.margins(0, .05)
#     plt.subplots_adjust(left = 0.1,bottom = 0.1,right=0.9,top=0.9,wspace=0.2,hspace=0.3) 
    


