#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct  6 15:35:43 2021

@author: wizard
"""

import h5py as imp
import numpy as np
import filtering.baseline as b
import filtering.plot_data as pdf
import filtering.filters as f

# Load the data
#file = imp.File(r"/home/wizard/Documents/Cerebro/Data/Intervention Group/Exp 13/Block 1/Set 1/saved_array_data.mat", "r")
file = imp.File(r"C:\Users\mbj\Desktop\ST7 Group\Data\Exp 15\Block 1\Set 1\saved_array_data.mat", "r")
data = file['Array_storage']['array_format_data'][46,:,4,1] #Always use number 2 of the last array in 4-d struct.
fs = file['Array_storage']['fs'][0,0]
t = np.arange(0,len(data),1)/fs


# Plot the raw data
pdf.plot_data_and_fft(data, t, fs,'Raw EEG signal',(0,60))


# Correct baseline
data_corrected = b.correct_baseline(data)
# Plot the baseline corrected
pdf.plot_data_and_fft(data_corrected, t, fs,'Baseline corrected EEG signal',(0,60))


# Bandpass filter to get spike
spike_250hz_10khz = f.filt_filt_but(data_corrected, 'bandpass', 4, (250,10000), fs)
# Plot the spike
pdf.plot_data_and_fft(spike_250hz_10khz, t, fs,'Spike between 250Hz and 10kHz', (250,10000))
pdf.zoom_plot(spike_250hz_10khz, t, 'Spike between 250Hz and 10kHz (500ms to 600ms)', (0.5,0.6))
pdf.zoom_plot(spike_250hz_10khz, t, 'Spike between 250Hz and 10kHz (500ms to 550ms)', (0.5,0.55))


# Keller et al. Low (0.3kHz to 2kHz) vs high MUA (8kHz to 10kHz) 
spike_300hz_2khz = f.filt_filt_but(data_corrected, 'bandpass', 4, (300,2000), fs)
spike_8khz_10khz = f.filt_filt_but(data_corrected, 'bandpass', 4, (8000,10000), fs)
spike_2khz_8khz = f.filt_filt_but(data_corrected, 'bandpass', 4, (2000,8000), fs)
# Plot the low MUA spike
pdf.plot_data_and_fft(spike_300hz_2khz, t, fs,'Low MUA (0.3kHz to 2kHz) Keller et al.', (300,2000))
pdf.zoom_plot(spike_300hz_2khz, t, 'Low MUA (0.3kHz to 2kHz) Keller et al. (500ms to 600ms)', (0.5,0.6))
pdf.zoom_plot(spike_300hz_2khz, t, 'Low MUA (0.3kHz to 2kHz) Keller et al. (500ms to 550ms)', (0.5,0.55))
# Middle MUA spike
pdf.plot_data_and_fft(spike_2khz_8khz, t, fs,'Middle MUA (2kHz to 8kHz) Keller et al.', (8000,10000))
pdf.zoom_plot(spike_2khz_8khz, t, 'Middle MUA (2kHz to 8kHz) Keller et al. (500ms to 600ms)', (0.5,0.6))
pdf.zoom_plot(spike_2khz_8khz, t, 'Middle MUA (2kHz to 8kHz) Keller et al. (500ms to 550ms)', (0.5,0.55))
# Plot the high MUA spike
pdf.plot_data_and_fft(spike_8khz_10khz, t, fs,'High MUA (8kHz to 10kHz) Keller et al.', (8000,10000))
pdf.zoom_plot(spike_8khz_10khz, t, 'High MUA (8kHz to 10kHz) Keller et al. (500ms to 600ms)', (0.5,0.6))
pdf.zoom_plot(spike_8khz_10khz, t, 'High MUA (8kHz to 10kHz) Keller et al. (500ms to 550ms)', (0.5,0.55))


# Belitski et al. 400Hz to 3kHz
spike_400hz_3khz = f.filt_filt_but(data_corrected, 'bandpass', 4, (400,3000), fs)
# Plot the spike
pdf.plot_data_and_fft(spike_400hz_3khz, t, fs,'(400Hz and 3kHz) Belitski et al.', (400,3000))
pdf.zoom_plot(spike_400hz_3khz, t, '(400Hz and 3kHz) Belitski et al. (500ms to 600ms)', (0.5,0.6))
pdf.zoom_plot(spike_400hz_3khz, t, '(400Hz and 3kHz) Belitski et al. (500ms to 550ms)', (0.5,0.55))
