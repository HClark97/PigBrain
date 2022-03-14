import tdt

data = tdt.read_block('Z:\Data\Pig Work\Suzan\Spring 2021\Student data\Subject9-210426-124955')
Fs = data.streams.RSn1.fs
data = data.streams.RSn1.data
import numpy as np
from scipy import signal
import scipy.fftpack 
from scipy.fft import fft, fftfreq
import matplotlib.pyplot as plt


f,pxx = scipy.signal.periodogram(data[1], fs=Fs, window='boxcar', nfft=None, detrend='constant', return_onesided=True, scaling='density', axis=- 1)

pxx = 10*np.log10(pxx)
plt.plot(f,pxx)
plt.grid()
plt.xlim(0,1000)
plt.xlabel("Frequency (Hz)")
plt.ylabel("Amplitude (dB/Hz)")
