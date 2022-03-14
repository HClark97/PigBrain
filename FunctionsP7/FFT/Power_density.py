# -*- coding: utf-8 -*-
"""
Created on Tue Oct  5 11:44:14 2021

@author: mbj
"""
import numpy as np
from scipy import signal
import scipy.fftpack 
from scipy.fft import fft, fftfreq
import matplotlib.pyplot as plt
def psd (data,Fs):
    """
    

    Parameters
    ----------
    data : Array of ints
        The data you would like to find the power spectrum density of.
    Fs : int/float
        The samplings frequency of the give data
    Returns
    -------
    The calculated power density of data, hwere f is the x-axis and rxx is the y-axis

    """
    f,pxx = scipy.signal.periodogram(data, fs=Fs, window='boxcar', nfft=None, detrend='constant', return_onesided=True, scaling='density', axis=- 1)
    pxx = 10*np.log10(pxx)
    return(f,pxx)


def psd_plot(data,Fs):
    """
    

    Parameters
    ----------
    data : Array of ints
        The data you would like to find the power spectrum density of.
    Fs : int/float
        The samplings frequency of the given data

    Returns
    -------
    plot

    """
    f,pxx = scipy.signal.periodogram(data, fs=Fs, window='boxcar', nfft=None, detrend='constant', return_onesided=True, scaling='density', axis=- 1)
    pxx = 10*np.log10(pxx)
    plt.plot(f,pxx)
    plt.grid()
    plt.xlim(0,100)
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Amplitude (dB/Hz)")
    
    
    
    
    
    
    
    
    
    
    
    
    