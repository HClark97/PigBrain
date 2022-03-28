# -*- coding: utf-8 -*-
"""
Created on Tue Oct  5 14:57:24 2021

@author: mbj
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import h5py as imp
import plotly.express as px
import plotly.io as pio
from scipy import signal
import matplotlib.pyplot as plt
import scipy.fftpack 
from scipy.fft import fft, fftfreq
import math

def psdmax (f,pxx,f1,f2):
    """

    Parameters
    ----------
    f : Array
        The frequency array you get after use of the psd function
    pxx : Array
        Power density values
    f1 :Int
        Start value from where the max value is found
    f2 : Int
        End value from where the max value is found

    Returns
    -------
    The max value of the array in the given interval

    """
    idx = np.where(np.logical_and(f >= f1, f <= f2))
    notchmax= np.max(pxx[idx])
    
    return(notchmax)

def psdmax_near(f,pxx,f1):
    """
    

    Parameters
    ----------
    f : Array
        The frequency array you get after use of the psd function
    pxx : Array
        Power density values
    f1 :Int
        Value for the frequency of interest

    Returns
    -------
    The max value of the array closest to the given value

    """
    f = np.asarray(f)
    idx = (np.abs(f - f1)).argmin()
    notchmax = np.max(pxx[idx])
    
    
    return(notchmax)