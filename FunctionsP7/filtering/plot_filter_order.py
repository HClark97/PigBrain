#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 11 09:42:56 2021

@author: amaliekoch
"""

"""
PLOTS DIFFERENT FILTER ORDERS (HIGHPASS)
"""

# lfp_high2 = f.filt_filt_but(data_base_corrected,'high', 1, 1, fs)
# lfp_high4 = f.filt_filt_but(data_base_corrected,'high', 2, 1, fs)
# lfp_high6 = f.filt_filt_but(data_base_corrected,'high', 3, 1, fs)
# lfp_high8 = f.filt_filt_but(data_base_corrected,'high', 4, 1, fs)
# lfp_high10 = f.filt_filt_but(data_base_corrected,'high', 5, 1, fs)
# lfp_high12= f.filt_filt_but(data_base_corrected,'high', 6, 1, fs)
# lfp_high14 = f.filt_filt_but(data_base_corrected,'high', 7, 1, fs)
# lfp_high16 = f.filt_filt_but(data_base_corrected,'high', 8, 1, fs)


# pdf.plot_data_and_fft(lfp_high2, t, fs, '2nd order filter',(0,7.5),(0,550),(-0.16,0.17))
# pdf.plot_data_and_fft(lfp_high4, t, fs, '4th order filter',(0,7.5),(0,550),(-0.16,0.17))
# pdf.plot_data_and_fft(lfp_high6, t, fs, '6th order filter',(0,7.5),(0,550),(-0.16,0.17))
# pdf.plot_data_and_fft(lfp_high8, t, fs, '8th order filter',(0,7.5),(0,550),(-0.16,0.17))
# pdf.plot_data_and_fft(lfp_high10, t, fs, '10th order filter',(0,7.5),(0,550),(-0.16,0.17))
# pdf.plot_data_and_fft(lfp_high12, t, fs, '12th order filter',(0,7.5),(0,550),(-0.16,0.17))
# pdf.plot_data_and_fft(lfp_high14, t, fs, '14th order filter',(0,7.5),(0,550),(-0.16,0.17))
# pdf.plot_data_and_fft(lfp_high16, t, fs, '16th order filter',(0,7.5),(0,550),(-0.16,0.17))

# #pdf.compare_plot_data_and_fft(lfp_high2, '2nd order', lfp_high16, '16th order', t, fs, '2nd order vs 16th order',(0,7.5),(0,550))
# pdf.compare_plot_data_and_fft(lfp_high2, '2nd order', lfp_high4, '4th order', t, fs, '2nd order vs 4th order',(0,7.5),(0,550))
# pdf.compare_plot_data_and_fft(lfp_high4, '4th order', lfp_high6, '6th order', t, fs, '4th order vs 6th order',(0,7.5),(0,550))
# pdf.compare_plot_data_and_fft(lfp_high6, '6th order', lfp_high8, '8th order', t, fs, '6th order vs 8th order',(0,7.5),(0,550))
# pdf.compare_plot_data_and_fft(lfp_high8, '8th order', lfp_high10, '10th order', t, fs, '8th order vs 10th order',(0,7.5),(0,550))
# pdf.compare_plot_data_and_fft(lfp_high10, '10th order', lfp_high12, '12th order', t, fs, '10th order vs 12th order',(0,7.5),(0,550))

"""
PLOTS DIFFERENT Q-FACTORS (NOTCH)
"""

# notch_q_25 = f.filt_filt_nocth(high_filtered,50, 25, fs)
# notch_q_50 = f.filt_filt_nocth(high_filtered,50, 50, fs)


# pdf.plot_data_and_fft(notch_q_25, t, fs, 'Q = 25',(40,60),(0,60))
# pdf.plot_data_and_fft(notch_q_50, t, fs, 'Q = 50',(40,60),(0,60))

# pdf.compare_plot_data_and_fft(notch_q_25, 'Q = 25', notch_q_50, 'Q = 50', t, fs, 'Q = 25 vs Q = 50',(40,60),(0,40))

# pdf.compare_plot_data_and_fft(high_filtered, 'High-pass filtered data', notch_q_25, 'Q = 25', t, fs, '50 Hz filtering using notch with Q = 25',(40,60),(0,60))
# pdf.compare_plot_data_and_fft(high_filtered, 'High-pass filtered data', notch_q_50, 'Q = 50', t, fs, '50 Hz filtering using notch with Q = 50',(40,60),(0,60))



"""
PLOTS DIFFERENT FILTER ORDERS (LOWPASS)
"""

# lfp_data1 = f.filt_filt_but(high_notch_filtered, 'low', 1, 250, fs)
# lfp_data2 = f.filt_filt_but(high_notch_filtered, 'low', 2, 250, fs)
# lfp_data3 = f.filt_filt_but(high_notch_filtered, 'low', 3, 250, fs)
# lfp_data4 = f.filt_filt_but(high_notch_filtered, 'low', 4, 250, fs)
# lfp_data5 = f.filt_filt_but(high_notch_filtered, 'low', 5, 250, fs)
# lfp_data6 = f.filt_filt_but(high_notch_filtered, 'low', 6, 250, fs)
# lfp_data7= f.filt_filt_but(high_notch_filtered, 'low', 7, 250, fs)
# lfp_data8= f.filt_filt_but(high_notch_filtered, 'low', 8, 250, fs)
# lfp_data9= f.filt_filt_but(high_notch_filtered, 'low', 9, 250, fs)
# lfp_data10= f.filt_filt_but(high_notch_filtered, 'low', 10, 250, fs)

# pdf.compare_plot_data_and_fft(lfp_data7, '14th order', lfp_data8, '16th order', t, fs, 'Low-pass filter cut-off 250 Hz: 14th order vs 16th order',(200,350),(0,15))

# pdf.compare_plot_data_and_fft(high_notch_filtered, 'Filtered data', lfp_data1, '2nd order low-pass filter', t, fs, 'Filterered data with 2nd order low-pass filter',(200,350),(0,15))
# pdf.compare_plot_data_and_fft(high_notch_filtered, 'Filtered data', lfp_data2, '4th order low-pass filter', t, fs, 'Filterered data with 4th order low-pass filter',(200,350),(0,15))
# pdf.compare_plot_data_and_fft(high_notch_filtered, 'Filtered data', lfp_data3, '6th order low-pass filter', t, fs, 'Filterered data with 6th order low-pass filter',(200,350),(0,15))
# pdf.compare_plot_data_and_fft(high_notch_filtered, 'Filtered data', lfp_data4, '8th order low-pass filter', t, fs, 'Filterered data with 8th order low-pass filter',(200,350),(0,15))
# pdf.compare_plot_data_and_fft(high_notch_filtered, 'Filtered data', lfp_data5, '10th order low-pass filter', t, fs, 'Filterered data with 10th order low-pass filter',(200,350),(0,15))
# pdf.compare_plot_data_and_fft(high_notch_filtered, 'Filtered data', lfp_data6, '12th order low-pass filter', t, fs, 'Filterered data with 12th order low-pass filter',(200,350),(0,15))
# pdf.compare_plot_data_and_fft(high_notch_filtered, 'Filtered data', lfp_data7, '14th order low-pass filter', t, fs, 'Filterered data with 14th order low-pass filter',(200,350),(0,15))
# pdf.compare_plot_data_and_fft(high_notch_filtered, 'Filtered data', lfp_data8, '16th order low-pass filter', t, fs, 'Filterered data with 16th order low-pass filter',(200,350),(0,15))
# pdf.compare_plot_data_and_fft(high_notch_filtered, 'Filtered data', lfp_data9, '18th order low-pass filter', t, fs, 'Filterered data with 18th order low-pass filter',(200,350),(0,15))
# pdf.compare_plot_data_and_fft(high_notch_filtered, 'Filtered data', lfp_data10, '20th order low-pass filter', t, fs, 'Filterered data with 20th order low-pass filter',(200,350),(0,15))


# pdf.compare_plot_data_and_fft(high_notch_filtered, 'Filtered data', lfp_data1, '2nd order low-pass filter', t, fs, 'Filterered data with 2nd order low-pass filter',(0.5,0.55),(-0.07,0.05),(200,350),(0,15))
# pdf.compare_plot_data_and_fft(high_notch_filtered, 'Filtered data', lfp_data2, '4th order low-pass filter', t, fs, 'Filterered data with 4th order low-pass filter',(0.5,0.55),(-0.07,0.05),(200,350),(0,15))
# pdf.compare_plot_data_and_fft(high_notch_filtered, 'Filtered data', lfp_data3, '6th order low-pass filter', t, fs, 'Filterered data with 6th order low-pass filter',(0.5,0.55),(-0.07,0.05),(200,350),(0,15))
# pdf.compare_plot_data_and_fft(high_notch_filtered, 'Filtered data', lfp_data4, '8th order low-pass filter', t, fs, 'Filterered data with 8th order low-pass filter',(0.5,0.55),(-0.07,0.05),(200,350),(0,15))
# pdf.compare_plot_data_and_fft(high_notch_filtered, 'Filtered data', lfp_data5, '10th order low-pass filter', t, fs, 'Filterered data with 10th order low-pass filter',(0.5,0.55),(-0.07,0.05),(200,350),(0,15))
# pdf.compare_plot_data_and_fft(high_notch_filtered, 'Filtered data', lfp_data6, '12th order low-pass filter', t, fs, 'Filterered data with 12th order low-pass filter',(0.5,0.55),(-0.07,0.05),(200,350),(0,15))
# pdf.compare_plot_data_and_fft(high_notch_filtered, 'Filtered data', lfp_data7, '14th order low-pass filter', t, fs, 'Filterered data with 14th order low-pass filter',(0.5,0.55),(-0.07,0.05),(200,350),(0,15))
# pdf.compare_plot_data_and_fft(high_notch_filtered, 'Filtered data', lfp_data8, '16th order low-pass filter', t, fs, 'Filterered data with 16th order low-pass filter',(0.5,0.55),(-0.07,0.05),(200,350),(0,15))
# pdf.compare_plot_data_and_fft(high_notch_filtered, 'Filtered data', lfp_data9, '18th order low-pass filter', t, fs, 'Filterered data with 18th order low-pass filter',(0.5,0.55),(-0.07,0.05),(200,350),(0,15))
# pdf.compare_plot_data_and_fft(high_notch_filtered, 'Filtered data', lfp_data10, '20th order low-pass filter', t, fs, 'Filterered data with 20th order low-pass filter',(0.5,0.55),(-0.07,0.05),(200,350),(0,15))



"""
PLOTS DIFFERENT FILTER ORDERS (BANDPASS)
"""

# spike_data6 = f.filt_filt_but(high_notch_filtered, 'bandpass', 3, (250,10000), fs)
# spike_data8 = f.filt_filt_but(high_notch_filtered, 'bandpass', 4, (250,10000), fs)
# spike_data10 = f.filt_filt_but(high_notch_filtered, 'bandpass', 5, (250,10000), fs)
# spike_data12 = f.filt_filt_but(high_notch_filtered, 'bandpass', 6, (250,10000), fs)
# spike_data14 = f.filt_filt_but(high_notch_filtered, 'bandpass', 7, (250,10000), fs)
# spike_data16 = f.filt_filt_but(high_notch_filtered, 'bandpass', 8, (250,10000), fs)
# spike_data18 = f.filt_filt_but(high_notch_filtered, 'bandpass', 9, (250,10000), fs)
# spike_data20 = f.filt_filt_but(high_notch_filtered, 'bandpass', 10, (250,10000), fs)
# spike_data28 = f.filt_filt_but(high_notch_filtered, 'bandpass', 14, (250,10000), fs)
# spike_data36 = f.filt_filt_but(high_notch_filtered, 'bandpass', 18, (250,10000), fs)
# spike_data44 = f.filt_filt_but(high_notch_filtered, 'bandpass', 22, (250,10000), fs)
# spike_data52 = f.filt_filt_but(high_notch_filtered, 'bandpass', 26, (250,10000), fs)
# spike_data60 = f.filt_filt_but(high_notch_filtered, 'bandpass', 30, (250,10000), fs)
# spike_data68 = f.filt_filt_but(high_notch_filtered, 'bandpass', 34, (250,10000), fs)
# spike_data76 = f.filt_filt_but(high_notch_filtered, 'bandpass', 38, (250,10000), fs)
# spike_data84 = f.filt_filt_but(high_notch_filtered, 'bandpass', 42, (250,10000), fs)
# spike_data92 = f.filt_filt_but(high_notch_filtered, 'bandpass', 46, (250,10000), fs)
# spike_data100 = f.filt_filt_but(high_notch_filtered, 'bandpass', 50, (250,10000), fs)



# pdf.compare_plot__bandpass_data_and_fft(high_notch_filtered, 'Filtered data',spike_data6 , 'Spike data', t, fs, 'Filtered data with 6th order bandpass filter cut-off: 250 Hz & 10000 Hz',(1.01,1.011),(-0.08,0.04),(200,300),(9500,10500),(0,20),(0,5))
# pdf.compare_plot__bandpass_data_and_fft(high_notch_filtered, 'Filtered data',spike_data8 , 'Spike data', t, fs, 'Filtered data with 8th order bandpass filter cut-off: 250 Hz & 10000 Hz',(1.01,1.011),(-0.08,0.04),(200,300),(9500,10500),(0,20),(0,5))
# pdf.compare_plot__bandpass_data_and_fft(high_notch_filtered, 'Filtered data',spike_data10 , 'Spike data', t, fs, 'Filtered data with 10th order bandpass filter cut-off: 250 Hz & 10000 Hz',(1.01,1.011),(-0.08,0.04),(200,300),(9500,10500),(0,20),(0,5))
# pdf.compare_plot__bandpass_data_and_fft(high_notch_filtered, 'Filtered data',spike_data12 , 'Spike data', t, fs, 'Filtered data with 12th order bandpass filter cut-off: 250 Hz & 10000 Hz',(1.01,1.011),(-0.08,0.04),(200,300),(9500,10500),(0,20),(0,5))
# pdf.compare_plot__bandpass_data_and_fft(high_notch_filtered, 'Filtered data',spike_data14 , 'Spike data', t, fs, 'Filtered data with 14th order bandpass filter cut-off: 250 Hz & 10000 Hz',(1.01,1.011),(-0.08,0.04),(200,300),(9500,10500),(0,20),(0,5))
# pdf.compare_plot__bandpass_data_and_fft(high_notch_filtered, 'Filtered data',spike_data16 , 'Spike data', t, fs, 'Filtered data with 16th order bandpass filter cut-off: 250 Hz & 10000 Hz',(1.01,1.011),(-0.08,0.04),(200,300),(9500,10500),(0,20),(0,5))
# pdf.compare_plot__bandpass_data_and_fft(high_notch_filtered, 'Filtered data',spike_data18 , 'Spike data', t, fs, 'Filtered data with 18th order bandpass filter cut-off: 250 Hz & 10000 Hz',(1.01,1.011),(-0.08,0.04),(200,300),(9500,10500),(0,20),(0,5))
# pdf.compare_plot__bandpass_data_and_fft(high_notch_filtered, 'Filtered data',spike_data20 , 'Spike data', t, fs, 'Filtered data with 20th order bandpass filter cut-off: 250 Hz & 10000 Hz',(1.01,1.011),(-0.08,0.04),(200,300),(9500,10500),(0,20),(0,5))
# pdf.compare_plot__bandpass_data_and_fft(high_notch_filtered, 'Filtered data',spike_data28 , 'Spike data', t, fs, 'Filtered data with 28th order bandpass filter cut-off: 250 Hz & 10000 Hz',(1.01,1.011),(-0.08,0.04),(200,300),(9500,10500),(0,20),(0,5))
# pdf.compare_plot__bandpass_data_and_fft(high_notch_filtered, 'Filtered data',spike_data36 , 'Spike data', t, fs, 'Filtered data with 36th order bandpass filter cut-off: 250 Hz & 10000 Hz',(1.01,1.011),(-0.08,0.04),(200,300),(9500,10500),(0,20),(0,5))
# pdf.compare_plot__bandpass_data_and_fft(high_notch_filtered, 'Filtered data',spike_data44 , 'Spike data', t, fs, 'Filtered data with 44th order bandpass filter cut-off: 250 Hz & 10000 Hz',(1.01,1.011),(-0.08,0.04),(200,300),(9500,10500),(0,20),(0,5))
# pdf.compare_plot__bandpass_data_and_fft(high_notch_filtered, 'Filtered data',spike_data52 , 'Spike data', t, fs, 'Filtered data with 52th order bandpass filter cut-off: 250 Hz & 10000 Hz',(1.01,1.011),(-0.08,0.04),(200,300),(9500,10500),(0,20),(0,5))
# pdf.compare_plot__bandpass_data_and_fft(high_notch_filtered, 'Filtered data',spike_data60 , 'Spike data', t, fs, 'Filtered data with 60th order bandpass filter cut-off: 250 Hz & 10000 Hz',(1.01,1.011),(-0.08,0.04),(200,300),(9500,10500),(0,20),(0,5))
# pdf.compare_plot__bandpass_data_and_fft(high_notch_filtered, 'Filtered data',spike_data68 , 'Spike data', t, fs, 'Filtered data with 68th order bandpass filter cut-off: 250 Hz & 10000 Hz',(1.01,1.011),(-0.08,0.04),(200,300),(9500,10500),(0,20),(0,5))
# pdf.compare_plot__bandpass_data_and_fft(high_notch_filtered, 'Filtered data',spike_data76 , 'Spike data', t, fs, 'Filtered data with 76th order bandpass filter cut-off: 250 Hz & 10000 Hz',(1.01,1.011),(-0.08,0.04),(200,300),(9500,10500),(0,20),(0,5))
# pdf.compare_plot__bandpass_data_and_fft(high_notch_filtered, 'Filtered data',spike_data84 , 'Spike data', t, fs, 'Filtered data with 84th order bandpass filter cut-off: 250 Hz & 10000 Hz',(1.01,1.011),(-0.08,0.04),(200,300),(9500,10500),(0,20),(0,5))
# pdf.compare_plot__bandpass_data_and_fft(high_notch_filtered, 'Filtered data',spike_data92 , 'Spike data', t, fs, 'Filtered data with 92th order bandpass filter cut-off: 250 Hz & 10000 Hz',(1.01,1.011),(-0.08,0.04),(200,300),(9500,10500),(0,20),(0,5))
# pdf.compare_plot__bandpass_data_and_fft(high_notch_filtered, 'Filtered data',spike_data100 , 'Spike data', t, fs, 'Filtered data with 100th order bandpass filter cut-off: 250 Hz & 10000 Hz',(1.01,1.011),(-0.08,0.04),(200,300),(9500,10500),(0,20),(0,5))

