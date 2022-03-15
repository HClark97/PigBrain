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
import 

fontsize_title = 25
fontsize_labels = 15

#load data
#data = tdt.read_block(r'C:\Users\mbj\Desktop\Uni\8. semester\Projekt\Data\Subject9-210426-124955')
data = tdt.read_block(r'C:\Users\mbj\Desktop\Uni\8. semester\Projekt\Data\Subject9-210526-093759')
#data = tdt.read_block(r'/Users/amaliekoch/Desktop/Subject1-210913-131914')

exp = 1
channel = 1

fs = data.streams.RSn1.fs
data = data.streams.RSn1.data

#data_test = data[:,50000:120000] #kun de første 12000 af data for hver kanal
data = data*1000; 

#data = data[:,50000:120000] #kun de første 12000 af data for hver kanal
t = np.arange(0,len(data[0,:]),1)/fs #time vector

#Plot PSD af raw data 
# pdf.psd_plot(data[1],fs,300)

p.plot_data_fft_log(data, t, fs, 'Raw data in time and frequency domain. ', channel, exp)


# #Harmonic notch filtering
harmonics = np.array([50,100,150,200,250,300,350])
notch_data = np.zeros(np.shape(data[0:8]))
notch_data = butfilt.filt_filt_nocth_harmonic(data,harmonics, fs)
# # HIGHPASS FILTERING 
high_notch_data = butfilt.filt_filt_but(notch_data, 'high', 5, 1, fs)
low_notch_data = butfilt.filt_filt_but(high_notch_data, 'low', 5, 250, fs)







# #plot PSD af notch harmonics filtering
# fig = plt.figure()
# pdf.psd_plot(notch_data[0],fs,400)
# plt.title("PSD after harmonics filtering",fontsize = fontsize_labels,fontweight='bold')


# #FFT og logaritmisk PLOT AF NOTCH FILTERING
# fig = plt.figure()
# data_fft = abs(rfft(notch_data[0]))
# freqs = rfftfreq(notch_data[0].size, 1 / fs)
# plt.semilogy(freqs, abs(data_fft))
# plt.grid(True)
# plt.grid(b = True, which = 'minor', color = 'k', linestyle = '-', alpha = 0.2)
# plt.minorticks_on()
# plt.title("Logarithmic frequency domain (after harmonics filtering)",fontsize = fontsize_labels,fontweight='bold')
# plt.ylabel('Power [µV^2]',fontsize = fontsize_labels,fontweight='bold')
# plt.xlabel('Frequency [Hz]',fontsize = fontsize_labels,fontweight='bold')

# fig = plt.figure()
# plt.plot(freqs, abs(data_fft))
# plt.title("FFT after harmonics filtering",fontsize = fontsize_labels,fontweight='bold')
# plt.xlim([-10,200])
# plt.ylim([0,600])

# # HIGHPASS FILTERING 
# high_notch_data = butfilt.filt_filt_but(notch_data[0], 'high', 5, 1, fs)
# high_notch_data_fft = abs(rfft(high_notch_data))
# freqs1 = rfftfreq(high_notch_data.size, 1 / fs)

# fig = plt.figure()
# plt.plot(freqs1, abs(high_notch_data_fft))
# plt.title("FFT (after harmonics and highpas filtering)",fontsize = fontsize_labels,fontweight='bold')
# plt.xlim([-10,200])
# plt.ylim([0,600])


for i in range (8):
    #PLOT AF RAW DATA I TID OG FREKVENS: 
    fig = plt.figure(figsize=(25, 20))
    plt.suptitle(('Raw data in time and frequency domain. ' + 'Channel: '+ str(i+1) + ' Exp ' + str(exp)),fontsize =25,fontweight='bold')
    plt.rc('xtick', labelsize=fontsize_labels)
    plt.rc('ytick', labelsize=fontsize_labels)
    plt.subplot(311)
    plt.plot(t,low_notch_data[i,:])
    #plt.xlim([0,2])  #only first part of data 
    #plt.ylim([-500,500])
    plt.grid(True)
    plt.title("Time domain",fontsize = fontsize_labels,fontweight='bold')
    plt.ylabel('Amplitude [µV]',fontsize = fontsize_labels,fontweight='bold')
    plt.xlabel('Time [s]',fontsize = fontsize_labels,fontweight='bold')
    plt.subplot(312)
    raw_data_fft = abs(rfft(low_notch_data[0]))
    freqs = rfftfreq(low_notch_data[0].size, 1 / fs)
    plt.plot(freqs, abs(raw_data_fft))
    plt.title("Frequency domain (FFT)",fontsize = fontsize_labels,fontweight='bold')
    plt.xlabel('Frequency [Hz]',fontsize = fontsize_labels,fontweight='bold')
    plt.ylabel('Amplitude',fontsize = fontsize_labels,fontweight='bold')
    #plt.ylim(-5,300)
    plt.xlim([-5,300])
    plt.subplot(313)
    plt.semilogy(freqs, abs(raw_data_fft**2))
    plt.grid(True)
    plt.grid(b = True, which = 'minor', color = 'k', linestyle = '-', alpha = 0.2)
    plt.minorticks_on()
    plt.title("Logarithmic frequency domain",fontsize = fontsize_labels,fontweight='bold')
    plt.ylabel('Power [µV^2]',fontsize = fontsize_labels,fontweight='bold')
    plt.xlabel('Frequency [Hz]',fontsize = fontsize_labels,fontweight='bold')
    #plt.ylim(-5,300)
    plt.xlim([-2,300])
    plt.margins(0, .05)
    plt.subplots_adjust(left = 0.1,bottom = 0.1,right=0.9,top=0.9,wspace=0.2,hspace=0.3) 
    
    


