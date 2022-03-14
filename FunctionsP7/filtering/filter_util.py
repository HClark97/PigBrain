#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct  1 12:43:27 2021

@author: amaliekoch
"""


def normalize_freq (crit_freq,fs):
     """
     Normalizing the critical frequency, so that 1 corresponds to nyquist (given as fs/2)

     Parameters
     ----------
     crit_freq : float
         critical/cutt-off frequency (in Hz) to be normalized.
     fs : float
         sampling frequency (in Hz)

     Returns
     -------
     The normalized frequency

     """
     return (crit_freq / ( fs / 2 ))
 
    

def denormalize_freq (norm_freq,fs):
     """
     Normalizing the critical frequency, so that 1 corresponds to nyquist (given as fs/2)

     Parameters
     ----------
     crit_freq : float
         critical/cutt-off frequency (in Hz) to be normalized.
     fs : float
         sampling frequency (in Hz)

     Returns
     -------
     The normalized frequency

     """
     return (norm_freq * ( fs / 2 ))
 
def compute_sample_time(fs):
    """
    Compute the sample time from the sample frequency.

    Parameters
    ----------
    fs : float
        The sample frequency in Hz.

    Returns
    -------
    float
        The sample time in seconds.

    """
    return ( 1 / ( fs * 2 ) )