# -*- coding: utf-8 -*-
import tdt
import numpy as np
from scipy import signal
import scipy.fftpack 
from scipy.fft import fft, fftfreq
import matplotlib.pyplot as plt
import FunctionsP7.FFT.Power_density as pdf
import FunctionsP7.filtering.filters as butfilt
import FunctionsP7.filtering.plot_data as pd
from numpy.fft import rfft, rfftfreq

fontsize_title = 25
fontsize_labels = 15

def plot_data_fft_log (data,t,fs,title, channel,exp):
    #PLOT AF RAW DATA I TID OG FREKVENS: 
    fig = plt.figure(figsize=(25, 20))
    plt.suptitle((title + ' - Channel: '+ str(channel) + ' Exp: ' + str(exp)),fontsize =25,fontweight='bold')
    plt.rc('xtick', labelsize=fontsize_labels)
    plt.rc('ytick', labelsize=fontsize_labels)
    plt.subplot(311)
    plt.plot(t,data[channel,:])
    plt.xlim([4,5])  #only first part of data 
    plt.ylim([15,20])
    plt.grid(True)
    plt.title("Time domain",fontsize = fontsize_labels,fontweight='bold')
    plt.ylabel('Amplitude [µV]',fontsize = fontsize_labels,fontweight='bold')
    plt.xlabel('Time [s]',fontsize = fontsize_labels,fontweight='bold')
    plt.subplot(312)
    raw_data_fft = abs(rfft(data[channel]))
    freqs = rfftfreq(data[channel].size, 1 / fs)
    plt.plot(freqs, abs(raw_data_fft))
    plt.title("Frequency domain (FFT)",fontsize = fontsize_labels,fontweight='bold')
    plt.xlabel('Frequency [Hz]',fontsize = fontsize_labels,fontweight='bold')
    plt.ylabel('Amplitude',fontsize = fontsize_labels,fontweight='bold')
    #plt.ylim(-5,300)
    plt.xlim([-5,350])
    plt.subplot(313)
    plt.semilogy(freqs, abs(raw_data_fft**2))
    plt.grid(True)
    plt.grid(b = True, which = 'minor', color = 'k', linestyle = '-', alpha = 0.2)
    plt.minorticks_on()
    plt.title("Logarithmic frequency domain",fontsize = fontsize_labels,fontweight='bold')
    plt.ylabel('Power [µV^2]',fontsize = fontsize_labels,fontweight='bold')
    plt.xlabel('Frequency [Hz]',fontsize = fontsize_labels,fontweight='bold')
    #plt.ylim(-5,300)
    plt.xlim([-5,350])
    plt.margins(0, .05)
    plt.subplots_adjust(left = 0.1,bottom = 0.1,right=0.9,top=0.9,wspace=0.2,hspace=0.3)



# t = np.arange(0,len(data[0,:]),1)/fs #time vector

# #Plot PSD af raw data 
# # pdf.psd_plot(data[1],fs,300)
# channel = 1

# p.plot_data_fft_log(data, t, fs, 'Raw data in time and frequency domain. ', channel, exp)

# # Harmonic notch filtering
# #harmonics = np.array([50,100,150,200,250,300,350])
# notch_data = np.zeros(np.shape(data))
# notch_data = butfilt.filt_filt_nocth(data, 50, 25, fs)
# #notch_data = butfilt.filt_filt_nocth(notch_data,50, 50/4, fs) #We choose Q-factor of 50/4 to get BW from 48 to 52 Hz

# p.plot_data_fft_log(notch_data, t, fs, 'Notch filtered data in time and frequency domain. ', channel, exp)


