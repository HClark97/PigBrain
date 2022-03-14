#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 11 09:45:29 2021

@author: amaliekoch

PLOTS DIFFERENT CUT_OFF FREQUENCIES (LFP DATA)

"""


# #Compare low-pass filtered data (250 cutoff) to the pre-processed data 
# pdf.compare_plot_data_and_fft(lfp_data1, 'LFP data - cutoff: 250 Hz',high_notch_filtered, 'pre-proc data', t, fs,'LFP data vs. pre-proccessed data',(0,400))
# pdf.compare_plot_data_and_fft(lfp_data1, 'LFP data - cutoff: 250 Hz',high_notch_filtered, 'pre-proc data', t, fs,'LFP data vs. pre-proccessed data - showing freq:200-400Hz',(200,400),(0,30))
# pdf.plot_data_and_fft(lfp_data1, t, fs, 'LFP',(0,260))

# #Compare 100Hz vs 250 Hz cutoff for low-pass filter
# pdf.compare_plot_data_and_fft(lfp_data1, '250 knæk',lfp_data4, '100 knæk', t, fs,'LFP data - low-pass filter cutoff 100 Hz vs 250 Hz',(0,250))
# pdf.compare_plot_data_and_fft(lfp_data1, '250 knæk',lfp_data4, '100 knæk', t, fs,'LFP data - low-pass filter cutoff 100 Hz vs 250 Hz',(90,250),(0,30))

# pdf.zoom_plot(lfp_data1, t, "zoom 250hz", (0.45,0.6))
# pdf.zoom_plot(lfp_data4, t, "zoom 100hz", (0.45,0.6))
# pdf.zoom_plot(lfp_data25, t, "zoom 175hz", (0.45,0.6))

# pdf.plot_data_and_fft(lfp_data1, t, fs, '2nd order filter',(0,300),(0,30))
# pdf.plot_data_and_fft(lfp_data2, t, fs, '4nd order filter',(0,300),(0,30))
# pdf.plot_data_and_fft(lfp_data3, t, fs, '6nd order filter',(0,300),(0,30))
# pdf.plot_data_and_fft(lfp_data4, t, fs, '8nd order filter',(0,300),(0,30))
# pdf.plot_data_and_fft(lfp_data5, t, fs, '10nd order filter',(0,300),(0,30))
# pdf.plot_data_and_fft(lfp_data6, t, fs, '12nd order filter',(0,300),(0,30))

# pdf.compare_plot_data_and_fft(lfp_data1, '2nd order', lfp_data6, '12nd order', t, fs, '2 vs 12',(0,300),(0,30))


#Compare raw epoch with filtered LFP epoch
#pdf.compare_plot_data_and_fft(lfp_data1, 'xx', data, 'raw', t, fs, "Raw data epoch vs LFP filtered epoch",(0,250))
