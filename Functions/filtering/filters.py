#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct  1 13:38:48 2021

@author: amaliekoch
"""

from scipy import signal

import numpy as np

def filt_filt_but(dataArray, filtType, order, critFreq, fs):
    """
    Apply a digital Butterworth filter to the data array using sosfiltfilt, with the given order and critical frquency.

    Parameters
    ----------
    dataArray : TYPE
        The data array to which the filter should be applied.
    filtType : string
        The filter type, as defined by scipy (‘lowpass’, ‘highpass’, ‘bandpass’, ‘bandstop’)
    order : integer
        The filter order.
    critFreq : float
        The critical frequency in Hz at which the filter attinuates the signal with -3dB.
    fs : float
        The sampling freqiency in Hz.
        
    Returns
    -------
    numpy array
        The filtered data array.
    """
    # Convert the critical frequency in Hz to Wn (a factor of fs)
    wn = critFreq / (fs / 2)
    # Design the filter
    sos = signal.butter(order, wn, btype=filtType, output='sos')
    
    
    # Apply the filter to the data array
    return signal.sosfiltfilt(sos, dataArray)


def filt_filt_butter(dataArray, filtType, order, critFreq, fs):
    """
    Apply a digital Butterworth filter to the data array using sosfiltfilt, with the given order and critical frquency.

    Parameters
    ----------
    dataArray : TYPE
        The data array to which the filter should be applied.
    filtType : string
        The filter type, as defined by scipy (‘lowpass’, ‘highpass’, ‘bandpass’, ‘bandstop’)
    order : integer
        The filter order.
    critFreq : float
        The critical frequency in Hz at which the filter attinuates the signal with -3dB.
    fs : float
        The sampling freqiency in Hz.
        
    Returns
    -------
    numpy array
        The filtered data array.
    """
    # Convert the critical frequency in Hz to Wn (a factor of fs)
    filtdata = dataArray
    wn = np.zeros(2)    
    for i in range (len(critFreq)):
        wn[0] = ((critFreq[i])-1) / (fs / 2)
        wn[1] = ((critFreq[i])+1) / (fs / 2)
        # Design the filter
        sos = signal.butter(order, wn, btype=filtType, output='sos')
        filtdata = signal.sosfiltfilt(sos, filtdata)#try
    return filtdata

def filt_filt_notch(data, crit_freq, Q, fs):
    """
    Filter a data array using a notch filter with the specified parameters.

    Parameters
    ----------
    data : array
        Data array to be filtered.
    crit_freq : float
        Critical frequency of the notch filter in Hz.
    Q : float
        Quality factor of the notch filter, used to calculate the bandwidth of
        the filter. See scipy.iirnotch documentation: Q = crit_freq / bw.
    fs : float
        The sampling freqiency in Hz.

    Returns
    -------
    array
        The filtered data array.
    """
    #Design notch filter 
    #f0 = 50.0  # Frequency to be removed from signal (Hz)
    #Q = 10.0  # Quality factor
    
    b_coef, a_coef = signal.iirnotch(crit_freq, Q, fs) #Design filter
    return signal.filtfilt(b_coef, a_coef, data) #apply filter

def filt_filt_nocth_harmonic(data, crit_freq, fs):
    """
    Filter a data array using a notch filter with the specified parameters.

    Parameters
    ----------
    data : array
        Data array to be filtered.
    crit_freq : float array
        Critical frequency of the notch filter in Hz.
    Q : float
        Quality factor of the notch filter, used to calculate the bandwidth of
        the filter. See scipy.iirnotch documentation: Q = crit_freq / bw.
    fs : float
        The sampling freqiency in Hz.

    Returns
    -------
    array
        The filtered data array.
    """
    
    #Design notch filter 
    #f0 = 50.0  # Frequency to be removed from signal (Hz)
    #Q = 10.0  # Quality factor
    filtdata = data
    
    for i in range (len(crit_freq)):
        Q = crit_freq[i]/2
        b_coef, a_coef = signal.iirnotch(crit_freq[i], Q, fs) #Design filter
        filtdata = signal.filtfilt(b_coef, a_coef, filtdata) #apply filter
    return filtdata

