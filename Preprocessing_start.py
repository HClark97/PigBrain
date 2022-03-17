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

#REMOVAL OF NOISY CHANNELS 

for group in range(np.size([Non,Nox])):
    for sets in range(np.shape(Nox['block'])):
        for channel in range(np.shape(data['NonnoxERP']['block'][0]['channel'])):
            for epoch in range(np.size(Nox['block'][0]['channel'][0]['ERPs'])):
                ## Baseline correction fra P7
                # baseline = np.mean(data[group,sets,channel,epoch]) #Taking the first 50 ms of the data for baseline correction for exp, channel and epoch
                # # Correct the baseline
                # data[group,sets,channel,epoch] = data[exp,channel,epoch] - baseline
                ## filter the data
                # harmonics = np.array([50,100,150,200,250,300,350])
                # data[group,sets,channel,epoch] = f.filt_filt_nocth_harmonic(data[group,sets,channel,epoch],harmonics, fs)
                # data[group,sets,channel,epoch] = f.filt_filt_but(data[group,sets,channel,epoch], 'band', 5, [1,250], fs) #10 order effective lowpass filter at 400 hz




