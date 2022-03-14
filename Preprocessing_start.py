import tdt
import numpy as np
from scipy import signal
import scipy.fftpack 
from scipy.fft import fft, fftfreq
import matplotlib.pyplot as plt
import FunctionsP7.FFT.Power_density as pdf
import FunctionsP7.filtering.filters as butfilt


#load data
data = tdt.read_block(r'C:\Users\mbj\Desktop\Uni\8. semester\Projekt\Data\Subject9-210426-124955')
notchdata = tdt.digitalfilter(data,'RSn1',np.array([10,100]),'NOTCH',np.array([50,100,150,200,250,300,350]))
fs = data.streams.RSn1.fs
data = data.streams.RSn1.data


#Plot PDF
pdf.psd_plot(data[1],fs,300)

harmonics = np.array([50,100,150,200,250,300,350])
#Notch filters
notch_data = np.zeros(np.shape(data))
notch_data = butfilt.filt_filt_nocth_harmonic(data,harmonics, fs)
fig = plt.figure()
pdf.psd_plot(notch_data[1],fs,300)





