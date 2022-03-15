import tdt
import numpy as np
from scipy import signal
import scipy.fftpack 
from scipy.fft import fft, fftfreq
import matplotlib.pyplot as plt
import FunctionsP7.FFT.Power_density as pdf
import FunctionsP7.filtering.filters as butfilt


#load data
data = tdt.read_block(r'C:\Users\mbj\Desktop\Uni\8. semester\Projekt\Data\Subject1-210914-103420')

#notchdata = tdt.digitalfilter(data,'RSn1',np.array([10,100]),'NOTCH',np.array([50,100,150,200,250,300,350]))
fs = data.streams.RSn1.fs
data = data.streams.RSn1.data

#Plot psd
fig2 = plt.figure()
pdf.psd_plot(data[2],fs,400)


#Harmonic notch filtering
harmonics = np.array([50,100,150,200,250,300,350])
notch_data = np.zeros([32,2])
notch_data = butfilt.filt_filt_nocth_harmonic(data[0:1],harmonics, fs)
fig3 = plt.figure()
pdf.psd_plot(notch_data[0], fs,400)

# If notch filter is to much, use the stopband filter below
# stop_data = butfilt.filt_filt_but_harmonic(data[0], 'stop', 1, harmonics, fs)
# fig = plt.figure()
# pdf.psd_plot(stop_data, fs,300)

#thrid try
#stop_data = butfilt.butter_bandstop_filter(data[0], 49, 51, fs, 5)
#fig = plt.figure()
#pdf.psd(stop_data, fs)
#Plot the results
# fig = plt.figure()
# pdf.psd_plot(notch_data[0],fs,400)







