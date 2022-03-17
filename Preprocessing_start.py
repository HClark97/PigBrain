import tdt
import numpy as np
from scipy import signal
import scipy.fftpack 
from scipy.fft import fft, fftfreq
import matplotlib.pyplot as plt
import FunctionsP7.FFT.Power_density as pdf
import FunctionsP7.filtering.filters as butfilt


#load data
data = tdt.read_block(r'C:\Users\mbj\Desktop\Uni\8. semester\Projekt\Data\Subject9-210526-093759')


fs = data.streams.RSn1.fs
data = data.streams.RSn1.data



#Plot psd
fig2 = plt.figure()
pdf.psd_plot(data[2],fs,400)

non/nox, sets,channel, stim/epoch
for nox in range(np.shape(files))
    for sets in range(np.shape(nox)
        for channel in range(np.shape(sets[0]))
            for epoch in range(np.shape(channel[0]))
## Baseline correction fra P7
# baseline = np.mean(data[exp,channel,epoch]) #Taking the first 50 ms of the data for baseline correction for exp, channel and epoch
# # Correct the baseline
# data[exp,channel,epoch] = data[exp,channel,epoch] - baseline
## filter the data
# harmonics = np.array([50,100,150,200,250,300,350])
# data[exp,channel,epoch] = f.filt_filt_nocth_harmonic(data[exp,channel,epoch],harmonics, fs)
# data[exp,channel,epoch] = f.filt_filt_but(data[exp,channel,epoch], 'band', 5, [1,250], fs) #10 order effective lowpass filter at 400 hz




