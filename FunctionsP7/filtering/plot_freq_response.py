#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct  1 11:05:36 2021

@author: amaliekoch
"""

from scipy import signal
import numpy as np
import matplotlib.pyplot as plt

import filtering.filter_util as fu
import filtering.plot_data as pd


"""
FILTER FREQUENCY RESPONSE PLOT (only gain(DB)):
    SÆT NEDENSTÅENDE I "FILTER.py" 
"""
# #Plot frequency response of highpass filter
# pfr.high_or_low_freq_resp_plot(1.3,0.6,8, fs, 'high',(0,10))

# #plot frequency response notch filter
# pfr.notch_freq_resp_plot(50, 10, fs)

# #plot frequency response lowpass filter
# pfr.high_or_low_freq_resp_plot(250,275,10, fs, 'low', (100, 400))
# #pfr.ellip_freq_resp_plot(250, 275, 3, 20,4, fs, 'low',(0,1000)) 

# #Plot frequency response of bandpass filter
# pfr.bandpass_freq_resp_plot(250,10000,50, fs,(-100,11000))
# pfr.bandpass_freq_resp_plot(250,10000,50, fs,(200,300))
# pfr.bandpass_freq_resp_plot(250,10000,50, fs,(9000,11000))


"""
FILTER FREQUENCY RESPONSE IN BOTH GAIN(DB) AND PHASE(RADIANS):
    SÆT NEDENSTÅENDE I "FILTER.py" 
"""
# #Plot frequency response of highpass filter
# pfr.high_or_low_freq_plot(1.3,6, fs, 'high')

# #plot frequency response notch filter
# pfr.notch_freq_plot(50, 10, fs)

# #plot frequency response lowpass filter
# pfr.high_or_low_freq_plot(250,50, fs, 'low', (230, 500))

# #Plot frequency response of bandpass filter
# pfr.bandpass_freq_plot(250, 10000, fs, 'Frequency response bandpass filter')
# pfr.bandpass_freq_plot(250,10000, fs, 'Frequency Response - bandpass filter',(200,300))
# pfr.bandpass_freq_plot(250,10000, fs, 'Frequency Response - bandpass filter',(9000,12000))


#TEST PLOT 
#pdf.test_plot_data_and_fft(data_base_corrected,data, t, fs,'Baseline corrected EEG signal',(0,60))
#pdf.compare_plot_data_and_fft(data_base_corrected, 'Baseline corrected EEG signal', data,'Raw EEG signal', t, fs, 'Baseline corrected EEG signal',(0,60))



#NEDENSTÅENDE PLOTTER FREQUENCY RESPONSE I GAIN(DB) (se længere nede for phase plot)
def bandpass_freq_resp_plot(start_freq,stop_freq,order,fs,freq_range, filename, save = False) :
    """
    Plot the frequency response of a bandpass filter with the given parameters.

    Parameters
    ----------
    start_freq : float
        The start frequency of the bandpass filter in Hz.
    stop_freq : float
        The stop frequency of the bandpass filter in Hz.
    fs : float
        The sampling frequency in Hz.
    title : string
        The title of the response plot.
    freq_range : (float,float), optional
        The limits on the x-axis in Hz. The default is none, which will plot 
        the entire response.

    Returns
    -------
    None.

    """
    wn_start = fu.normalize_freq(start_freq, fs) #normaliseret frequency (start)
    wn_stop = fu.normalize_freq(stop_freq, fs) #normaliseret frequency (stop)
    
    start_limit = 0
    stop_limit = 0

    if freq_range is not None:
        start_limit = freq_range[0]
        stop_limit = freq_range[1]
    else:
        stop_limit = fs/2

    if start_limit<0:
        start_limit = 0
    if stop_limit>fs/2: 
        stop_limit = fs/2
    
    wornN = np.arange(0,wn_stop+0.1,(1 / (fs * 2)))
    sos = signal.butter(order, [wn_start, wn_stop], 'bandpass', analog=False, output='sos', fs=fs)
    w, h = signal.sosfreqz(sos, worN=wornN, fs=fs)
    
    x_freqz = fu.denormalize_freq(w, fs)
    
    fig = plt.figure(figsize=(15, 10))
    #plt.subplot(2, 1, 1)
    db = 20*np.log10(np.maximum(np.abs(h), 1e-5))
    #plt.plot(w, db)
    plt.plot(x_freqz,db, color=pd.colors['signal'][0])
    plt.rc('xtick', labelsize= pd.fontsize_labels)
    plt.rc('ytick', labelsize= pd.fontsize_labels)    
    plt.grid(True)
    plt.yticks([0,-10, -20,-30, -40, -50,-60,-70,-80,-90,-100])
    plt.ylabel('Gain [dB]',fontsize = pd.fontsize_labels,fontweight='bold')
    plt.xlabel('Frequency [Hz]',fontsize = pd.fontsize_labels,fontweight='bold')
    plt.title('Frequency Response - ' + str(order) + 'th order ' + 'band-pass filter',fontsize = pd.fontsize_labels,fontweight='bold')
    plt.axvline(fu.denormalize_freq(wn_start, fs),label='Passband start frequency: ' + str(start_freq)+' Hz', color = pd.colors['cut-off'])
    plt.xlim(start_limit, stop_limit)
    #plt.axvline(wn_stop, color='red')
    plt.axvline(fu.denormalize_freq(wn_stop, fs),label='Passband stop frequency: ' +str(stop_freq)+' Hz', color = pd.colors['cut-off'])
    plt.legend()
    if save:
        fig.savefig('{}.eps'.format(filename),format='eps')
   


def high_or_low_freq_resp_plot(pass_freq,stop_freq, order, fs, filt_type, freq_range, filename, save = False):
    """
    Plot the frequency response of a vlowpass or highpass filter with the 
     given parameters.

    Parameters
    ----------
    crit_freq : float
        The critical frequency of the filter in Hz.
    order : integer
        The filter order.
    fs : float
        The sampling frequency in Hz.
    filt_type : string
        Filter type as either 'low' for a lowpass filter or 'high' for a high
        pass filter.
    freq_range : (float, float), optional
        Range to plot on the x-axis in Hz. Should be above 0 Hz and below fs/2.
        The default is None, which plots only +/- 10 Hz from the critical
        frequency.

    Returns
    -------
    None.

    """
    
     
    
    wn = fu.normalize_freq(pass_freq, fs) #normaliseret frequency
    start_limit = pass_freq-20
    stop_limit = pass_freq+20

    if freq_range is not None:
        start_limit = freq_range[0]
        stop_limit = freq_range[1]

    if start_limit < 0:
        start_limit = 0
    if stop_limit > fs/2: 
        stop_limit = fs/2
    
    wornN = np.arange(
        fu.normalize_freq(start_limit, fs),
        fu.normalize_freq(stop_limit, fs),
        fu.compute_sample_time(fs)
    )
    sos = signal.butter(order, wn, filt_type, analog=False, output='sos', fs=fs)
    w, h = signal.sosfreqz(sos, worN=wornN, fs=fs)
    
    x_freq = fu.denormalize_freq(w, fs)
    
    fig = plt.figure(figsize=(15, 10))
    db = 20*np.log10(np.maximum(np.abs(h), 1e-5))
    plt.plot(x_freq, db, color=pd.colors['signal'][0])
    plt.grid(True)
    plt.rc('xtick', labelsize= pd.fontsize_labels)
    plt.rc('ytick', labelsize= pd.fontsize_labels)    
    plt.yticks([0,-10, -20,-30, -40, -50,-60,-70,-80,-90,-100])
    plt.ylabel('Gain [dB]',fontsize = pd.fontsize_labels,fontweight='bold')
    plt.xlabel('Frequency [Hz]',fontsize = pd.fontsize_labels,fontweight='bold')
    plt.title('Frequency Response - ' + str(order) + 'th order ' + filt_type + '-pass filter',fontsize = pd.fontsize_labels,fontweight='bold')
    plt.axvline(fu.denormalize_freq(wn, fs),label='Passband frequency: ' + str(pass_freq) + ' Hz', color = pd.colors['cut-off'])
    
    
    if stop_freq is not None: 
         wn_stop = fu.normalize_freq(stop_freq, fs) #normaliseret frequency
         plt.axvline(fu.denormalize_freq(wn_stop, fs), color='red',label='Stopband frequency: ' + str(stop_freq) + ' Hz')
    
    plt.legend()
    plt.xlim(start_limit, stop_limit)
    if save:
        fig.savefig('{}.eps'.format(filename),format='eps')

def ellip_freq_resp_plot(pass_freq,stop_freq,rip_pas,rip_stop,order, fs, filt_type, freq_range=None):
    """
    Plot the frequency response of a vlowpass or highpass filter with the 
     given parameters.

    Parameters
    ----------
    crit_freq : float
        The critical frequency of the filter in Hz.
    order : integer
        The filter order.
    fs : float
        The sampling frequency in Hz.
    filt_type : string
        Filter type as either 'low' for a lowpass filter or 'high' for a high
        pass filter.
    freq_range : (float, float), optional
        Range to plot on the x-axis in Hz. Should be above 0 Hz and below fs/2.
        The default is None, which plots only +/- 10 Hz from the critical
        frequency.

    Returns
    -------
    None.

    """
    wn = fu.normalize_freq(pass_freq, fs) #normaliseret frequency
    wn_stop = fu.normalize_freq(stop_freq, fs) #normaliseret frequency
    start_limit = pass_freq-20
    stop_limit = pass_freq+20

    if freq_range is not None:
        start_limit = freq_range[0]
        stop_limit = freq_range[1]

    if start_limit < 0:
        start_limit = 0
    if stop_limit > fs/2: 
        stop_limit = fs/2
    
    wornN = np.arange(
        fu.normalize_freq(start_limit, fs),
        fu.normalize_freq(stop_limit, fs),
        fu.compute_sample_time(fs)
    )
    
    sos = signal.ellip(order, rip_pas,rip_stop,wn, filt_type, analog=False,output='sos', fs=fs)
    w, h = signal.sosfreqz(sos, worN=wornN, fs=fs)
    
    x_freq = fu.denormalize_freq(w, fs)
    
    plt.figure()
    db = 20*np.log10(np.maximum(np.abs(h), 1e-5))
    plt.plot(x_freq, db)
    plt.grid(True)
    plt.yticks([0,-10, -20,-30, -40, -50,-60,-70,-80,-90,-100])
    plt.ylabel('Gain [dB]')
    plt.xlabel('Frequency [Hz]')
    plt.title('Frequency Response - ' + str(order) + 'th order ' + filt_type + '-pass filter')
    plt.axvline(fu.denormalize_freq(wn, fs), color='green',label='Passband frequency: ' + str(pass_freq) + ' Hz')
    plt.axvline(fu.denormalize_freq(wn_stop, fs), color='red',label='Stopband frequency: ' + str(stop_freq) + ' Hz')
    plt.legend()
    plt.xlim(start_limit, stop_limit)
    
def notch_freq_resp_plot(crit_freq, Q, fs, filename, save = False):
    """
    Display the frequency and phase response of an IIR notch filter

    Parameters
    ----------
    crit_freq : float
        The critical frequency in Hz.
    Q : float
        The filter quality factor. Defines the length of the passband Q=crit_freq/filter_bandwidth. See doc for scipy.signal.iirnotch.
    fs : float
        The sampling frequency.

    Returns
    -------
    None.

    """
    # Design the filter
    b_coef, a_coef = signal.iirnotch(crit_freq, Q, fs) #Design the notch filter
    # Calculate the frequency range area of interest to be plotted
    start_limit = crit_freq - 5 # Calculate the frequency range to be plotted
    if start_limit < 0: # Avoid expanding outside frequency range
        start_limit = 0
    stop_limit = crit_freq + 5
    if stop_limit > fs/2: 
        stop_limit = fs/2
    worN = np.arange(start_limit, stop_limit, fu.compute_sample_time(fs)) # Create array of frequencies of interest
    # Perform the frequency analysis
    x_freq, h = signal.freqz(b_coef, a_coef, worN=worN, fs=fs)
    # Create and display the plot
    fig = plt.figure(figsize=(15, 10)) # Create the figure
    # Create the attinuation plot
    #plt.subplot(2, 1, 1) # Create the subplot
    plt.title('Frequency Response - Notch Filter', fontsize = pd.fontsize_labels,fontweight='bold')
    db = 20 * np.log10( np.maximum( np.abs( h ), 1e-5)) # Calculate the attinuation in dB
    plt.plot(x_freq, db) # Plot the filter response
    plt.grid(True) # Display grid
    plt.yticks([0,-10, -20,-30, -40, -50,-60,-70,-80,-90]) # Set ticks on y-axis
    plt.xlim(start_limit, stop_limit) # Limit the x-axis to the area of interest
    plt.ylabel('Gain [dB]', fontsize = pd.fontsize_labels, fontweight='bold')
    plt.xlabel('Frequency [Hz]', fontsize = pd.fontsize_labels, fontweight='bold')
    plt.rc('xtick', labelsize = pd.fontsize_labels)
    plt.rc('ytick', labelsize = pd.fontsize_labels)   
    plt.axvline(crit_freq, label='Cut-off frequency: ' + str(crit_freq)+' Hz, Q-factor = ' + str(Q), color = pd.colors['cut-off'])
    plt.legend() # Enable legends
    
    if save:
        fig.savefig('{}.eps'.format(filename),format='eps')   


#NEDENSTÅENDE PLOTTER FREQUENCY RESPONSE I BÅDE GAIN(DB) og PHASE (RADIANS)

def bandpass_freq_plot(start_freq,stop_freq,fs,title,freq_range=None) :
    """
    Plot the frequency response of a bandpass filter with the given parameters.

    Parameters
    ----------
    start_freq : float
        The start frequency of the bandpass filter in Hz.
    stop_freq : float
        The stop frequency of the bandpass filter in Hz.
    fs : float
        The sampling frequency in Hz.
    title : string
        The title of the response plot.
    freq_range : (float,float), optional
        The limits on the x-axis in Hz. The default is none, which will plot 
        the entire response.

    Returns
    -------
    None.

    """
    wn_start = fu.normalize_freq(start_freq, fs) #normaliseret frequency (start)
    wn_stop = fu.normalize_freq(stop_freq, fs) #normaliseret frequency (stop)
    
    start_limit = 0
    stop_limit = 0

    if freq_range is not None:
        start_limit = freq_range[0]
        stop_limit = freq_range[1]
    else:
        stop_limit = fs/2

    if start_limit<0:
        start_limit = 0
    if stop_limit>fs/2: 
        stop_limit = fs/2
    
    wornN = np.arange(0,wn_stop+0.1,(1 / (fs * 2)))
    sos = signal.butter(50, [wn_start, wn_stop], 'bandpass', analog=False, output='sos', fs=fs)
    w, h = signal.sosfreqz(sos, worN=wornN, fs=fs)
    
    x_freqz = fu.denormalize_freq(w, fs)
    
    plt.figure()
    plt.subplot(2, 1, 1)
    db = 20*np.log10(np.maximum(np.abs(h), 1e-5))
    #plt.plot(w, db)
    plt.plot(x_freqz,db)
    plt.grid(True)
    plt.yticks([0, -20, -40, -60])
    plt.ylabel('Gain [dB]')
    plt.title(title)
    plt.axvline(fu.denormalize_freq(wn_start, fs), color='red',label='Cut-off frequency')
    plt.legend()
    plt.xlim(start_limit, stop_limit)
    #plt.axvline(wn_stop, color='red')
    plt.axvline(fu.denormalize_freq(wn_stop, fs), color='red')
    plt.subplot(2, 1, 2)
    
    #plt.plot(w, np.angle(h))
    plt.plot(x_freqz, np.angle(h))
    plt.grid(True)
    plt.yticks([-np.pi, -0.5*np.pi, 0, 0.5*np.pi, np.pi],[r'$-\pi$', r'$-\pi/2$', '0', r'$\pi/2$', r'$\pi$'])
    plt.ylabel('Phase [rad]')
    plt.xlabel('Frequency [Hz]')
    plt.xlim(start_limit, stop_limit)


def high_or_low_freq_plot(crit_freq, order, fs, filt_type, freq_range=None):
    """
    Plot the frequency response of a vlowpass or highpass filter with the 
     given parameters.

    Parameters
    ----------
    crit_freq : float
        The critical frequency of the filter in Hz.
    order : integer
        The filter order.
    fs : float
        The sampling frequency in Hz.
    filt_type : string
        Filter type as either 'low' for a lowpass filter or 'high' for a high
        pass filter.
    freq_range : (float, float), optional
        Range to plot on the x-axis in Hz. Should be above 0 Hz and below fs/2.
        The default is None, which plots only +/- 10 Hz from the critical
        frequency.

    Returns
    -------
    None.

    """
    wn = fu.normalize_freq(crit_freq, fs) #normaliseret frequency
    start_limit = crit_freq-20
    stop_limit = crit_freq+20

    if freq_range is not None:
        start_limit = freq_range[0]
        stop_limit = freq_range[1]

    if start_limit < 0:
        start_limit = 0
    if stop_limit > fs/2: 
        stop_limit = fs/2
    
    wornN = np.arange(
        fu.normalize_freq(start_limit, fs),
        fu.normalize_freq(stop_limit, fs),
        fu.compute_sample_time(fs)
    )
    sos = signal.butter(order, wn, filt_type, analog=False, output='sos', fs=fs)
    w, h = signal.sosfreqz(sos, worN=wornN, fs=fs)
    
    x_freq = fu.denormalize_freq(w, fs)
    
    plt.figure()
    plt.subplot(2, 1, 1)
    db = 20*np.log10(np.maximum(np.abs(h), 1e-5))
    plt.plot(x_freq, db)
    plt.grid(True)
    plt.yticks([0, -20, -40, -60])
    plt.ylabel('Gain [dB]')
    plt.title('Frequency Response ' + filt_type + 'pass filter')
    plt.axvline(fu.denormalize_freq(wn, fs), color='red',label='Cut-off frequency')
    plt.legend()
    plt.xlim(start_limit, stop_limit)
    plt.subplot(2, 1, 2)
    
    plt.plot(x_freq, np.angle(h))
    plt.grid(True)
    plt.yticks([-np.pi, -0.5*np.pi, 0, 0.5*np.pi, np.pi],[r'$-\pi$', r'$-\pi/2$', '0', r'$\pi/2$', r'$\pi$'])
    plt.ylabel('Phase [rad]')
    plt.xlabel('Frequency [Hz]')
    plt.xlim(start_limit, stop_limit)
    
def notch_freq_plot(crit_freq, Q, fs):
    """
    Display the frequency and phase response of an IIR notch filter

    Parameters
    ----------
    crit_freq : float
        The critical frequency in Hz.
    Q : float
        The filter quality factor. Defines the length of the passband Q=crit_freq/filter_bandwidth. See doc for scipy.signal.iirnotch.
    fs : float
        The sampling frequency.

    Returns
    -------
    None.

    """
    # Design the filter
    b_coef, a_coef = signal.iirnotch(crit_freq, Q, fs) #Design the notch filter
    # Calculate the frequency range area of interest to be plotted
    start_limit = crit_freq - 10 # Calculate the frequency range to be plotted
    if start_limit < 0: # Avoid expanding outside frequency range
        start_limit = 0
    stop_limit = crit_freq + 10
    if stop_limit > fs/2: 
        stop_limit = fs/2
    worN = np.arange(start_limit, stop_limit, fu.compute_sample_time(fs)) # Create array of frequencies of interest
    # Perform the frequency analysis
    x_freq, h = signal.freqz(b_coef, a_coef, worN=worN, fs=fs)
    # Create and display the plot
    plt.figure() # Create the figure
    # Create the attinuation plot
    plt.subplot(2, 1, 1) # Create the subplot
    plt.title('Frequency Response notch filter') # Set the plot title
    db = 20 * np.log10( np.maximum( np.abs( h ), 1e-5)) # Calculate the attinuation in dB
    plt.plot(x_freq, db) # Plot the filter response
    plt.grid(True) # Display grid
    plt.yticks([0, -20, -40, -60]) # Set ticks on y-axis
    plt.xlim(start_limit, stop_limit) # Limit the x-axis to the area of interest
    plt.ylabel('Gain [dB]') # Axis labels
    plt.axvline(crit_freq, color = 'red', label = 'Cut-off frequency') # Display a line at the critical frequency
    plt.legend() # Enable legends
    # Create the phase plot
    plt.subplot(2, 1, 2)
    plt.plot(x_freq, np.angle( h )) # Plot the phase
    plt.grid(True)
    plt.yticks([-np.pi, -0.5*np.pi, 0, 0.5*np.pi, np.pi],[r'$-\pi$', r'$-\pi/2$', '0', r'$\pi/2$', r'$\pi$'])
    plt.ylabel('Phase [rad]')
    plt.xlabel('Frequency [Hz]')
    plt.xlim(start_limit, stop_limit)
