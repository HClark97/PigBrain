#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct  1 12:55:32 2021

@author: amaliekoch
"""
from scipy import signal
import numpy as np
import matplotlib.pyplot as plt
from numpy.fft import rfft, rfftfreq

colors = {
    'text': 'black',
    'cut-off': 'red',
    'signal': ['royalblue','forestgreen','darkorange','palevioletred','sienna', 'hotpink','plum','cornflowerblue','blue','cyan','mediumturquoise','springgreen','forestgreen','darkseagreen','yellow','khaki','gold','darkorange','peachpuff','sienna','red']
}
fontsize_title = 25
fontsize_labels = 15

def plot_data_and_fft (data,t,fs,title,filename,x_range_fft = None,y_range_fft = None,x_range_data = None,y_range_data= None,save = False):
    """
    Plots a data array and the FFT of the data array.

    Parameters
    ----------
    data : array
        Data array to be plotted and FFT'ed.
    t : array
        The time axis of the data array, using seconds as unit (1/fs).
    fs : float
        The sampling freqiency, in Hz.
    title : string
        The plot title to be displayed.
    x_range_fft : (float, float), optional
        The limits on the x-axis. The default is None.
    y_range_fft : (float, flaot), optional
        The limits on the y-axis. The default is None.

    Returns
    -------
    None.

    """
        
    data= data*1000
    
    if y_range_data is None:
        y_range_data=(min(data),max(data))
        
    if x_range_data is None:
        x_range_data=(0,t[-1])

    fig = plt.figure(figsize=(15, 10))
    plt.suptitle(title,fontsize =25,fontweight='bold')
    plt.rc('xtick', labelsize=fontsize_labels)
    plt.rc('ytick', labelsize=fontsize_labels)
    plt.subplot(211)
    plt.plot(t, data,color = colors['signal'][0])
    plt.xlim(x_range_data)
    plt.ylim(y_range_data)
    plt.grid(True)
    plt.title("Time domain",fontsize = fontsize_labels,fontweight='bold')
    plt.ylabel('Amplitude [µV]',fontsize = fontsize_labels,fontweight='bold')
    plt.xlabel('Time [s]',fontsize = fontsize_labels,fontweight='bold')
    plt.subplot(212)
    data_fft = abs(rfft(data))**2
    freqs = rfftfreq(data.size, 1 / fs)
    plt.semilogy(freqs, abs(data_fft),color = colors['signal'][0])
    plt.grid(True)
    plt.grid(b = True, which = 'minor', color = 'k', linestyle = '-', alpha = 0.2)
    plt.minorticks_on()
    plt.title("Logarithmic frequency domain",fontsize = fontsize_labels,fontweight='bold')
    plt.ylabel('Power [µV^2]',fontsize = fontsize_labels,fontweight='bold')
    plt.xlabel('Frequency [Hz]',fontsize = fontsize_labels,fontweight='bold')
    
    if x_range_fft is None:
        x_range_fft=(0,(fs/2))
        
    if y_range_fft is None:
        y_range_fft=(0,max(abs(data_fft)))

    plt.ylim(y_range_fft)
    plt.xlim(x_range_fft)
    plt.margins(0, .05)
    plt.subplots_adjust(left = 0.1,bottom = 0.1,right=0.9,top=0.9,wspace=0.2,hspace=0.3)
    if save:
        fig.savefig('{}.eps'.format(filename),format='pdf')
    
def plot_3times_data_and_fft (data1,data2,data3,t,fs,title,x_range_fft = None,y_range_fft= None):
    """
    Plots three data arrays and the associated FFT of each data array.

    Parameters
    ----------
    data1 : array
        Data array 1 to be plotted and FFT'ed.
    data2 : array
        Data array 2 to be plotted and FFT'ed.
    data3 : array
        Data array 3 to be plotted and FFT'ed.
    t : array
        The time axis of the data array, using seconds as unit (1/fs).
    fs : float
        The sampling freqiency, in Hz.
    title : string
        The plot title to be displayed.
    x_range_fft : (float, float), optional
        The limits on the x-axis. The default is None.
    y_range_fft : (float, flaot), optional
        The limits on the y-axis. The default is None.

    Returns
    -------
    None.

    """
    data1 = data1*1000
    data2 = data2*1000
    data3 = data3*1000
    
    plt.figure(figsize=(15, 10))
    plt.suptitle(title,fontsize = 20,color='Black')
    
    #Plot raw data 
    plt.subplot(321)
    plt.plot(t, data1)
    plt.xlim(0,t[-1])
    plt.grid(True)
    plt.title("Raw signal")
    plt.ylabel('Amplitude [µV]')
    #plt.xlabel('Time [s]')
    plt.margins(0.1)
    
    #Plot FFT of raw data
    plt.subplot(322)
    data_fft = rfft(data1)
    freqs = rfftfreq(data1.size, 1 / fs)
    plt.plot(freqs, abs(data_fft))
    plt.grid(True)
    plt.title("FFT of raw signal")
    plt.ylabel('Amplitude [µV]')
    #plt.xlabel('Frequency [Hz]')
    plt.margins(0.1,0.1)
    
    if x_range_fft is None:
        x_range_fft=(0,(fs/2))
        
    if y_range_fft is None:
        y_range_fft=(0,max(data_fft))
    
    plt.ylim(y_range_fft)
    plt.xlim(x_range_fft)
    #plt.margins(0, .05)
    
    #Plot highpass filtered signal
    plt.subplot(323)
    plt.plot(t, data2)
    plt.xlim(0,t[-1])
    plt.grid(True)
    plt.title("Highpass filtered signal")
    plt.ylabel('Amplitude [µV]')
    #plt.xlabel('Time [s]')
    plt.margins(0.1,0.1)
    
    #Plot FFT of highpass filtered signal
    plt.subplot(324)
    filtered_fft = rfft(data2)
    plt.plot(freqs, abs(filtered_fft))
    plt.ylim(y_range_fft)
    plt.xlim(x_range_fft)
    plt.grid(True)
    plt.title("FFT of highpass filtered signal")
    plt.ylabel('Amplitude [µV]')
    #plt.xlabel('Frequency [Hz]')
    plt.margins(0.1,0.1)
    
    #Plot highpass + notch filtered signal
    plt.subplot(325)
    plt.plot(t, data3)
    plt.grid(True)
    plt.xlim(0,t[-1])
    plt.title("Highpass and notch filtered signal")
    plt.ylabel('Amplitude [µV]')
    plt.xlabel('Time [s]')
    plt.margins(0.1,0.1)
    
    #Plot FFT highpass + notch filtered signal
    plt.subplot(326)
    filtered_fft = rfft(data3)
    plt.plot(freqs, abs(filtered_fft))
    plt.ylim(y_range_fft)
    plt.xlim(x_range_fft)
    plt.grid(True)
    plt.title("FFT of highpas and notch filtered signal")
    plt.ylabel('Amplitude [µV]')
    plt.xlabel('Frequency [Hz]')
    plt.margins(0.1,0.1)
    

def plot_preProcessing (raw_data,base_correct_data,high_data,high_notch_data,t,fs,title,filename, x_range_fft = None,y_range_fft= None,save=False):
    """
    Plots the four steps of our preprocessing: Raw signal, baseline correction,
    highpass filtering and notch filtering.

    Parameters
    ----------
    raw_data : array
        The raw EEG signal.
    base_correct_data : array
        The base line corrected EEG signal.
    high_data : array
        The high pass filtered and baseline corrected EEG signal.
    high_notch_data : array
        The notch and high pass filtered and baseline corrected EEG signal.
    t : array
        The time axis of the data array, using seconds as unit (1/fs).
    fs : float
        The sampling freqiency, in Hz.
    title : string
        The plot title to be displayed.
    x_range_fft : (float, float), optional
        The limits on the x-axis. The default is None.
    y_range_fft : (float, flaot), optional
        The limits on the y-axis. The default is None.


    Returns
    -------
    None.

    """
    data1 = raw_data*1000
    data2 = base_correct_data*1000
    data3 = high_data*1000
    data4 = high_notch_data*1000
    
    fig = plt.figure(figsize=(15, 10))
    plt.tight_layout(h_pad=2)
    plt.suptitle(title,fontsize = 25,fontweight='bold')
    
    fontsize_labels = 15
    
    #Plot raw data 
    #plt.subplot(4,2,(1,3))
    plt.subplot(421)
    plt.plot(t, data1,label='Raw signal',color = colors['signal'][0])
    plt.xlim(0,t[-1])
    plt.grid(True)
    plt.title("Raw signal",fontsize = fontsize_labels,fontweight='bold')
    plt.ylabel('Amplitude [µV]',fontsize = fontsize_labels,fontweight='bold')
    plt.legend(fontsize = 10)
    #plt.xlabel('Time [s]')
    

    #Plot FFT of raw data
    plt.subplot(422)
    data_fft = abs(rfft(data1))**2
    freqs = rfftfreq(data1.size, 1 / fs)
    plt.semilogy(freqs, abs(data_fft),color = colors['signal'][0])
    plt.grid(True)
    plt.grid(b = True, which = 'minor', color = 'k', linestyle = '-', alpha = 0.2)
    plt.minorticks_on()
    plt.title("Log frequency plot of raw signal",fontsize = fontsize_labels,fontweight='bold')
    plt.ylabel('Power [µV^2]',fontsize = fontsize_labels,fontweight='bold')
    #plt.xlabel('Frequency [Hz]')
    
    if x_range_fft is None:
        x_range_fft=(0,(fs/2))
        
    if y_range_fft is None:
        y_range_fft=(0,max(data_fft))
    
    plt.ylim(y_range_fft)
    plt.xlim(x_range_fft)
    
    #Plot 
    #plt.subplot(4,2,(1,3))
    plt.subplot(423)
    plt.plot(t, data1,label='Raw signal',color = colors['signal'][0])
    plt.plot(t, data2, color = colors['signal'][1],label='Baseline corrected signal')
    plt.legend(fontsize = 10)
    plt.xlim(0,t[-1])
    plt.grid(True)
    plt.title("Baseline corrected signal",fontsize = fontsize_labels,fontweight='bold')
    plt.ylabel('Amplitude [µV]',fontsize = fontsize_labels,fontweight='bold')
    #plt.xlabel('Time [s]')
    
    #Plot FFT of baseline corrected data 
    plt.subplot(424)
    baseCorrect_data_fft = abs(rfft(data2))**2
    freqs = rfftfreq(data2.size, 1 / fs)
    plt.semilogy(freqs, abs(baseCorrect_data_fft),color = colors['signal'][1])
    plt.grid(True)
    plt.grid(b = True, which = 'minor', color = 'k', linestyle = '-', alpha = 0.2)
    plt.minorticks_on()
    plt.ylim(y_range_fft)
    plt.xlim(x_range_fft)
    plt.title("Log frequency plot of baseline corrected signal",fontsize = fontsize_labels,fontweight='bold')
    plt.ylabel('Power [µV^2]',fontsize = fontsize_labels,fontweight='bold')
    #plt.xlabel('Frequency [Hz]')
    
    #Plot highpass filtered signal
    plt.subplot(425)
    plt.plot(t, data2,color = colors['signal'][1],label='Baseline corrected signal' )
    plt.plot(t, data3, color = colors['signal'][2], label='Highpass filtered signal')
    plt.legend(fontsize = 10)
    plt.xlim(0,t[-1])
    plt.grid(True)
    plt.title("Highpass filtered signal",fontsize = fontsize_labels,fontweight='bold')
    plt.ylabel('Amplitude [µV]',fontsize = fontsize_labels,fontweight='bold')
    #plt.xlabel('Time [s]')
    
    #Plot FFT of highpass filtered signal
    plt.subplot(426)
    high_filtered_fft = abs(rfft(data3))**2
    plt.semilogy(freqs, abs(high_filtered_fft),color = colors['signal'][2])
    plt.grid(b = True, which = 'minor', color = 'k', linestyle = '-', alpha = 0.2)
    plt.minorticks_on()
    plt.ylim(y_range_fft)
    plt.xlim(x_range_fft)
    plt.grid(True)
    plt.title("Log frequency plot of highpass filtered signal",fontsize = fontsize_labels,fontweight='bold')
    plt.ylabel('Power [µV^2]',fontsize = fontsize_labels,fontweight='bold')
    #plt.xlabel('Frequency [Hz]')
    
    #Plot highpass + notch filtered signal
    plt.subplot(427)
    plt.plot(t, data3,color = colors['signal'][2], label='Highpass filtered signal')
    plt.plot(t, data4, color = colors['signal'][3],label='Highpass and notch filtered signal')
    plt.legend(fontsize = 10)
    plt.grid(True)
    plt.xlim(0,t[-1])
    plt.title("Highpass and notch filtered signal",fontsize = fontsize_labels,fontweight='bold')
    plt.ylabel('Amplitude [µV]',fontsize = fontsize_labels,fontweight='bold')
    plt.xlabel('Time [s]',fontsize = fontsize_labels,fontweight='bold')
    
    #Plot FFT highpass + notch filtered signal
    plt.subplot(428)
    high_notch_filtered_fft = abs(rfft(data4))**2
    plt.semilogy(freqs, abs(high_notch_filtered_fft),color = colors['signal'][3])
    plt.grid(b = True, which = 'minor', color = 'k', linestyle = '-', alpha = 0.2)
    plt.minorticks_on()
    plt.ylim(y_range_fft)
    plt.xlim(x_range_fft)
    plt.grid(True)
    plt.title("Log frequency plot of notch filtered signal",fontsize = fontsize_labels,fontweight='bold')
    plt.ylabel('Power [µV^2]',fontsize = fontsize_labels,fontweight='bold')
    plt.xlabel('Frequency [Hz]',fontsize = fontsize_labels,fontweight='bold')
    
    plt.subplots_adjust(left = 0.1,bottom = 0.1,right=0.9,top=0.9,wspace=0.2,hspace=0.4)
    if save:
        fig.savefig('{}.eps'.format(filename),format='pdf')
    
    
def compare_plot_data_and_fft (data,
                               data_label,
                               compareData,
                               compareData_label,
                               t,
                               fs,
                               title,
                               filename,
                               x_range_time = None,
                               y_range_time = None,
                               x_range_fft = None,
                               y_range_fft= None,
                               cut_off_frequency = None,
                               save = False):
    """
    

    Parameters
    ----------
    data : TYPE
        DESCRIPTION.
    data_label : TYPE
        DESCRIPTION.
    compareData : TYPE
        DESCRIPTION.
    compareData_label : TYPE
        DESCRIPTION.
    t : TYPE
        DESCRIPTION.
    fs : TYPE
        DESCRIPTION.
    title : TYPE
        DESCRIPTION.
    x_range_time : TYPE, optional
        DESCRIPTION. The default is None.
    y_range_time : TYPE, optional
        DESCRIPTION. The default is None.
    x_range_fft : TYPE, optional
        DESCRIPTION. The default is None.
    y_range_fft : TYPE, optional
        DESCRIPTION. The default is None.
    cut_off_frequency : TYPE, optional
        DESCRIPTION. The default is None.

    Returns
    -------
    None.

    """
    
    
    data= data*1000
    compareData= compareData*1000
    if x_range_time is None:
        x_range_time = (0,t[-1])
        
    if y_range_fft is None:
        y_range_fft=(min(min(data),min(compareData)),max(max(data),max(compareData)))
        
    fontsize_labels = 15
    
    fig = plt.figure(figsize=(15, 10))
    plt.suptitle(title,fontsize = 25,fontweight='bold')
    plt.rc('xtick', labelsize=fontsize_labels)
    plt.rc('ytick', labelsize=fontsize_labels)
    plt.subplot(211)
    plt.plot(t, data, label = data_label,color = colors['signal'][0])
    plt.plot(t,compareData,color = colors['signal'][1], label = compareData_label) #linewidth=5.0
    plt.xlim(x_range_time)
    plt.ylim(y_range_time)
    plt.legend(fontsize =13)
    plt.grid(True)
    plt.title("Time domain",fontsize = fontsize_labels,fontweight='bold')
    plt.ylabel('Amplitude [µV]',fontsize = fontsize_labels,fontweight='bold')
    plt.xlabel('Time [s]',fontsize = fontsize_labels,fontweight='bold')
    plt.subplot(212)
    data_fft = abs(rfft(data))**2
    data_fft1 = abs(rfft(compareData))**2
    freqs = rfftfreq(data.size, 1 / fs)
    freqs1 = rfftfreq(compareData.size, 1 / fs)
    plt.semilogy(freqs, (abs(data_fft)),label = data_label,color = colors['signal'][0])
    plt.semilogy(freqs1,(abs(data_fft1)),color = colors['signal'][1],label = compareData_label)
    plt.grid(True)
    plt.grid(b = True, which = 'minor', color = 'k', linestyle = '-', alpha = 0.2)
    plt.minorticks_on()
    if cut_off_frequency is not None:
        plt.axvline(cut_off_frequency, color = colors['cut-off'],label='Cut-off frequency: {} Hz'.format(cut_off_frequency))
    
    plt.legend(fontsize =13)
    plt.grid(True)
    plt.title("Logarithmic frequency domain",fontsize = fontsize_labels,fontweight='bold')
    plt.ylabel('Power [µV^2]',fontsize = fontsize_labels,fontweight='bold')
    plt.xlabel('Frequency [Hz]',fontsize = fontsize_labels,fontweight='bold')

    
    if x_range_fft is None:
        x_range_fft=(0,(fs/2))
        
    if y_range_fft is None:
        y_range_fft=(0,max(abs(data_fft)))
    
    plt.ylim(y_range_fft)
    plt.xlim(x_range_fft)
    plt.margins(0, .05)
    plt.subplots_adjust(left = 0.1,bottom = 0.1,right=0.9,top=0.9,wspace=0.2,hspace=0.3)
    if save:
        fig.savefig('{}.eps'.format(filename),format='pdf')
    
def zoom_plot(data, t, title, xlim=None, ylim=None):
    data= data*1000
    plt.figure(figsize=(15, 10))
    plt.suptitle(title,fontsize = 25,color='Black')
    plt.plot(t, data)
    
    if xlim is None:
        xlim=(0,len(data))
        
    if ylim is None:
        ylim=(min(data),max(data))
    
    plt.xlim(xlim)
    plt.ylim(ylim)
    plt.grid(True)
    plt.title("Time domain",fontsize = 15)
    plt.ylabel('Amplitude [µV]')
    plt.xlabel('Time [s]')

def compare_plot__bandpass_data_and_fft (data,
                                         data_label,
                                         compareData,
                                         compareData_label,
                                         t,
                                         fs,
                                         title,
                                         x_range_time = None,
                                         y_range_time = None,
                                         x_range_fft1 = None,
                                         x_range_fft2 = None,
                                         y_range_fft1= None,
                                         y_range_fft2= None,
                                         cut_off_frequencies = (None, None),
                                         save = False):
    """
    

    Parameters
    ----------
    data : TYPE
        DESCRIPTION.
    data_label : TYPE
        DESCRIPTION.
    compareData : TYPE
        DESCRIPTION.
    compareData_label : TYPE
        DESCRIPTION.
    t : TYPE
        DESCRIPTION.
    fs : TYPE
        DESCRIPTION.
    title : TYPE
        DESCRIPTION.
    x_range_time : TYPE, optional
        DESCRIPTION. The default is None.
    y_range_time : TYPE, optional
        DESCRIPTION. The default is None.
    x_range_fft1 : TYPE, optional
        DESCRIPTION. The default is None.
    x_range_fft2 : TYPE, optional
        DESCRIPTION. The default is None.
    y_range_fft1 : TYPE, optional
        DESCRIPTION. The default is None.
    y_range_fft2 : TYPE, optional
        DESCRIPTION. The default is None.
    save : TYPE, optional
        DESCRIPTION. The default is False.

    Returns
    -------
    None.

    """
    
    data= data*1000
    compareData= compareData*1000
    
    fig = plt.figure(figsize=(15, 10))
    plt.suptitle(title,fontsize = 25,color='Black')
    plt.subplot(2,2,(1,2))
    plt.plot(t, data, label = data_label)
    plt.plot(t,compareData,color = 'green', label = compareData_label)
    plt.legend()
    plt.xlim(x_range_time)
    plt.ylim(y_range_time)
    plt.grid(True)
    plt.title("Time domain",fontsize = 15)
    plt.ylabel('Amplitude [µV]')
    plt.xlabel('Time [s]')
    
    plt.subplot(2,2,3)
    data_fft = rfft(data)
    data_fft1 = rfft(compareData)
    freqs = rfftfreq(data.size, 1 / fs)
    freqs1 = rfftfreq(compareData.size, 1 / fs)
    plt.plot(freqs, abs(data_fft), label = data_label)
    plt.plot(freqs1, abs(data_fft1), color = 'green', label = compareData_label)
    plt.grid(True)
    plt.title("Low frequency domain",fontsize = 15)
    plt.ylabel('Amplitude [µV]')
    plt.xlabel('Frequency [Hz]')
    if cut_off_frequencies[0] is not None:
        plt.axvline(cut_off_frequencies[0], color='red',label='Pass-band start frequency: {} Hz'.format(cut_off_frequencies[0]))
    plt.legend()
    
    if x_range_fft1 is None:
        x_range_fft1=(0,(fs/2))
        
    if x_range_fft2 is None:
        x_range_fft2=(0,(fs/2))    
        
    if y_range_fft1 is None:
        y_range_fft1=(0,max(abs(data_fft)))
    
    if y_range_fft2 is None:
        y_range_fft2=(0,max(abs(data_fft)))
    
    plt.ylim(y_range_fft1)
    plt.xlim(x_range_fft1)
    
    plt.subplot(2,2,4)
    data_fft = rfft(data)
    data_fft1 = rfft(compareData)
    freqs = rfftfreq(data.size, 1 / fs)
    freqs1 = rfftfreq(compareData.size, 1 / fs)
    plt.plot(freqs, abs(data_fft),label = data_label)
    plt.plot(freqs1, abs(data_fft1),color = 'green',label = compareData_label)
    plt.grid(True)
    plt.title("High frequency domain",fontsize = 15)
    plt.ylabel('Amplitude [µV]')
    plt.xlabel('Frequency [Hz]')
    if cut_off_frequencies[1] is not None:
        plt.axvline(cut_off_frequencies[1], color='red',label='Pass-band stop frequency: {} Hz'.format(cut_off_frequencies[1]))
    plt.ylim(y_range_fft2)
    plt.xlim(x_range_fft2)
    plt.legend()
    plt.margins(0, .05)
    plt.subplots_adjust(left = 0.1,bottom = 0.1,right=0.9,top=0.9,wspace=0.2,hspace=0.3)
    
    if save:
        fig.savefig('bandpass_compare_plot_{}.eps'.format(title.replace(' ', '')),format='eps')

def compare_3plot_bandpass_data_and_fft (data,
                                         data_label,
                                         compareData1,
                                         compareData1_label,
                                         compareData2,
                                         compareData2_label,
                                         t,
                                         fs,
                                         title,
                                         x_range_time = None,
                                         y_range_time = None,
                                         x_range_fft1 = None,
                                         x_range_fft2 = None,
                                         y_range_fft1= None,
                                         y_range_fft2= None,
                                         cut_off_frequencies = (None, None),
                                         save = False):
    """
    

    Parameters
    ----------
    data : TYPE
        DESCRIPTION.
    data_label : TYPE
        DESCRIPTION.
    compareData1 : TYPE
        DESCRIPTION.
    compareData1_label : TYPE
        DESCRIPTION.
    compareData2 : TYPE
        DESCRIPTION.
    compareData2_label : TYPE
        DESCRIPTION.
    t : TYPE
        DESCRIPTION.
    fs : TYPE
        DESCRIPTION.
    title : TYPE
        DESCRIPTION.
    x_range_time : TYPE, optional
        DESCRIPTION. The default is None.
    y_range_time : TYPE, optional
        DESCRIPTION. The default is None.
    x_range_fft1 : TYPE, optional
        DESCRIPTION. The default is None.
    x_range_fft2 : TYPE, optional
        DESCRIPTION. The default is None.
    y_range_fft1 : TYPE, optional
        DESCRIPTION. The default is None.
    y_range_fft2 : TYPE, optional
        DESCRIPTION. The default is None.
    cut_off_frequencies : TYPE, optional
        DESCRIPTION. The default is (None, None).
    save : TYPE, optional
        DESCRIPTION. The default is False.

    Returns
    -------
    None.

    """
    
    data = data * 1000
    compareData1 = compareData1 * 1000
    compareData2 = compareData2 * 1000
    
    fig = plt.figure(figsize=(15, 10))
    plt.suptitle(title,fontsize = 25,color='Black')
    plt.subplot(2,2,(1,2))
    plt.plot(t, data, label = data_label)
    plt.plot(t,compareData1,color = 'green', label = compareData1_label)
    plt.plot(t,compareData2,color = 'orange', label = compareData2_label)
    plt.legend()
    plt.xlim(x_range_time)
    plt.ylim(y_range_time)
    plt.grid(True)
    plt.title("Time domain",fontsize = 15)
    plt.ylabel('Amplitude [µV]')
    plt.xlabel('Time [s]')
    
    plt.subplot(2,2,3)
    data_fft = rfft(data)
    data_fft1 = rfft(compareData1)
    data_fft2 = rfft(compareData2)
    freqs = rfftfreq(data.size, 1 / fs)
    freqs1 = rfftfreq(compareData1.size, 1 / fs)
    freqs2 = rfftfreq(compareData2.size, 1 / fs)
    plt.plot(freqs, abs(data_fft), label = data_label)
    plt.plot(freqs1, abs(data_fft1), color = 'green', label = compareData1_label)
    plt.plot(freqs2, abs(data_fft2), color = 'orange', label = compareData2_label)
    plt.grid(True)
    plt.title("Low frequency domain",fontsize = 15)
    plt.ylabel('Amplitude [µV]')
    plt.xlabel('Frequency [Hz]')
    if cut_off_frequencies[0] is not None:
        plt.axvline(cut_off_frequencies[0], color='red',label='Pass-band start frequency: {} Hz'.format(cut_off_frequencies[0]))
    plt.legend()
    
    if x_range_fft1 is None:
        x_range_fft1=(0,(fs/2))
        
    if x_range_fft2 is None:
        x_range_fft2=(0,(fs/2))    
        
    if y_range_fft1 is None:
        y_range_fft1=(0,max(abs(data_fft)))
    
    if y_range_fft2 is None:
        y_range_fft2=(0,max(abs(data_fft)))
    
    plt.ylim(y_range_fft1)
    plt.xlim(x_range_fft1)
    
    plt.subplot(2,2,4)
    data_fft = rfft(data)
    data_fft1 = rfft(compareData1)
    data_fft2 = rfft(compareData2)
    freqs = rfftfreq(data.size, 1 / fs)
    freqs1 = rfftfreq(compareData1.size, 1 / fs)
    freqs2 = rfftfreq(compareData2.size, 1 / fs)
    plt.plot(freqs, abs(data_fft),label = data_label)
    plt.plot(freqs1, abs(data_fft1),color = 'green',label = compareData1_label)
    plt.plot(freqs2, abs(data_fft2),color = 'orange',label = compareData2_label)
    plt.grid(True)
    plt.title("High frequency domain",fontsize = 15)
    plt.ylabel('Amplitude [µV]')
    plt.xlabel('Frequency [Hz]')
    if cut_off_frequencies[1] is not None:
        plt.axvline(cut_off_frequencies[1], color='red',label='Pass-band stop frequency: {} Hz'.format(cut_off_frequencies[1]))
    plt.ylim(y_range_fft2)
    plt.xlim(x_range_fft2)
    plt.legend()
    plt.margins(0, .05)
    plt.subplots_adjust(left = 0.1,bottom = 0.1,right=0.9,top=0.9,wspace=0.2,hspace=0.3)
    
    if save:
        fig.savefig('bandpass_compare_plot_{}.eps'.format(title.replace(' ', '')),format='eps')


def compare_3_data_and_fft (data,data_label,compareData,compareData_label,compareData2,compareData2_label,t,fs,title,filename,x_range_time = None,y_range_time = None,x_range_fft = None,y_range_fft= None, cut_off_frequency=None,save=None):
    
    data= data*1000
    compareData= compareData*1000
    compareData2= compareData2*1000
    if x_range_time is None:
        x_range_time = (0,t[-1])
        
    if y_range_fft is None:
        y_range_fft=(min(min(data),min(compareData)),max(max(data),max(compareData)))
    
    fontsize_labels = 20
    fig = plt.figure(figsize=(15, 10))
    plt.suptitle(title,fontsize = 25,fontweight='bold')
    plt.rc('xtick', labelsize=fontsize_labels)
    plt.rc('ytick', labelsize=fontsize_labels)
    plt.subplot(211)
    plt.plot(t, data,color = colors['signal'][0], label = data_label)
    plt.plot(t,compareData2,color = colors['signal'][2], label = compareData2_label)
    plt.plot(t,compareData,color = colors['signal'][1], label = compareData_label)
    plt.xlim(x_range_time)
    plt.ylim(y_range_time)
    plt.legend(fontsize =15)
    plt.grid(True)

    plt.title("Time domain",fontsize = fontsize_labels,fontweight='bold')
    plt.ylabel('Amplitude [µV]',fontsize = fontsize_labels,fontweight='bold')
    plt.xlabel('Time [s]',fontsize = fontsize_labels,fontweight='bold')
    plt.subplot(212)
    data_fft = abs(rfft(data))**2
    data_fft1 = abs(rfft(compareData))**2
    data_fft2 = abs(rfft(compareData2))**2
    
    freqs = rfftfreq(data.size, 1 / fs)
    freqs1 = rfftfreq(compareData.size, 1 / fs)
    freqs2 = rfftfreq(compareData2.size, 1 / fs)
    
    if cut_off_frequency is not None:
        plt.axvline(cut_off_frequency, color = colors['cut-off'],label='Cut-off frequency: {} Hz'.format(cut_off_frequency))
    

    if x_range_fft is None:
        x_range_fft=(0,(fs/2))
        
    if y_range_fft is None:
        y_range_fft=(0,max(abs(data_fft)))
    
    plt.semilogy(freqs, abs(data_fft),color = colors['signal'][0],label = data_label)
    plt.semilogy(freqs1, abs(data_fft2),color = colors['signal'][2],label = compareData_label)
    plt.semilogy(freqs2, abs(data_fft1),color = colors['signal'][1],label = compareData2_label)
    plt.grid(b = True, which = 'minor', color = 'k', linestyle = '-', alpha = 0.2)
    plt.minorticks_on()
    plt.legend(fontsize =10)
    plt.grid(True)
    plt.title("Logarithmic frequency domain",fontsize = fontsize_labels,fontweight='bold')
    plt.ylabel('Power [µV^2]',fontsize = fontsize_labels,fontweight='bold')
    plt.xlabel('Frequency [Hz]',fontsize = fontsize_labels,fontweight='bold')
    
    plt.ylim(y_range_fft)
    plt.xlim(x_range_fft)
    plt.margins(0, .05)
    plt.subplots_adjust(left = 0.1,bottom = 0.1,right=0.9,top=0.9,wspace=0.2,hspace=0.3)
    if save:
        fig.savefig('{}.eps'.format(filename),format='pdf')

##############################################################################
###                               4x4 Plots                                ###
##############################################################################
def plot_4x4_bad_channels(data, fs, x_ticks, y_ticks, title, filename, save = False):
    """
    Create a 4x4 plot of the data from 12 set.

    Parameters
    ----------
    data : numpy.array(12,:,16)
        A numpy array with 12 sets in the first entry, the datapoints in the 
        second entry and the 16 channels in the third entry.
    fs : TYPE
        DESCRIPTION.
    y_ticks : TYPE
        DESCRIPTION.
    x_ticks : TYPE
        DESCRIPTION.
    title : TYPE
        DESCRIPTION.
    save : TYPE, optional
        DESCRIPTION. The default is False.

    Returns
    -------
    None.

    """
    plot_4x4_of_sets(data, 1000, fs, x_ticks, y_ticks, 'Time [s]', 'Amplitude [μV]', title, filename, save)

def plot_4x4_normalized(data, fs, x_ticks, y_ticks, title, filename, save = False):
    """
    

    Parameters
    ----------
    data : TYPE
        DESCRIPTION.
    fs : TYPE
        DESCRIPTION.
    x_ticks : TYPE
        DESCRIPTION.
    y_ticks : TYPE
        DESCRIPTION.
    title : TYPE
        DESCRIPTION.
    save : TYPE, optional
        DESCRIPTION. The default is False.

    Returns
    -------
    None.

    """
    plot_4x4_of_sets(data, None, fs, x_ticks, y_ticks, 'Time [s]', 'Z-score [σ]', title, filename, save)
    

def plot_4x4_of_sets(data, 
                     scale_data,
                     fs, 
                     x_ticks, 
                     y_ticks, 
                     x_label,
                     y_label,
                     title,
                     filename,
                     save = False):
    # Scale the data
    if scale_data:
        data = data * scale_data
    # Scale the y_ticks if no ticks are given
    if y_ticks == None:
        x_tick_start = round(x_ticks[0] * fs)
        x_tick_end = round(x_ticks[-1] * fs)
        minVal = round(np.nanmin(data[:,x_tick_start:x_tick_end,:]) - 0.05, 2)
        maxVal = round(np.nanmax(data[:,x_tick_start:x_tick_end,:]) + 0.05, 2)
        diff = maxVal - minVal
        step = round(diff/4,2)
        y_ticks = np.zeros(5,dtype="float32")
        for tick in range(0,5,1):
            y_ticks[tick] = round(minVal + tick * step,2)
    # Create the figure
    fig = plt.figure(figsize=(15, 10))
    # Create a title
    plt.suptitle(title, fontsize = 25, fontweight='bold', color=colors['text'])
    # Create a time axis
    t = np.arange(0,len(data[0,:,0]),1)/fs
    # Create a channel index to plot the channels in the correct order
    # Loop through each subplot index
    for sp_index in range(1,17):
        # Create the subplot and select the index
        plt.subplot(4,4,sp_index)
        channel_index = sp_index-1
        # Loop through each set
        for sets in range(0,np.shape(data)[0]):
            # Plot the set data
            plt.plot(t, data[sets,:,channel_index], color=colors['signal'][sets])
            # Set axis limits
            plt.xlim(x_ticks[0], x_ticks[-1])
            plt.ylim(y_ticks[0], y_ticks[-1])
            # Set the channel number
            plt.title('Channel ' + str(channel_index + 1), fontsize = fontsize_labels, fontweight='bold')
            # Set frid
            plt.grid(True)
            # If the subplot index is on the left axis, set y-axis
            if sp_index in [1,5,9,13]:
                plt.yticks(y_ticks)
                plt.ylabel(y_label, fontsize = fontsize_labels, fontweight='bold')
                plt.rc('ytick', labelsize=fontsize_labels)
            else:
                # Unset the y axis label
                plt.yticks(y_ticks, labels = "")
            if sp_index in [13,14,15,16]:
                plt.xticks(x_ticks)
                plt.xlabel(x_label, fontsize = fontsize_labels, fontweight='bold')
                plt.rc('xtick', labelsize=fontsize_labels)
            else:
                 plt.xticks(x_ticks, labels = "")
    # Export the figure if set
    if save:
        fig.savefig('{}.eps'.format(filename),format='pdf')

# Spike plots




def plot_4x4_of_spikes(data, 
                     scale_data,
                     fs, 
                     x_ticks, 
                     y_ticks, 
                     x_label,
                     y_label,
                     title,
                     filename,
                     save = False):
    # Allocate RMS
    RMS = np.zeros(16)
    # Scale the data
    if scale_data:
        data = data * scale_data
    # Scale the y_ticks if no ticks are given
    if y_ticks == None:
        x_tick_start = round(x_ticks[0] * fs)
        x_tick_end = round(x_ticks[-1] * fs)
        minVal = round(np.nanmin(data[:,x_tick_start:x_tick_end,:]) - 0.05, 2)
        maxVal = round(np.nanmax(data[:,x_tick_start:x_tick_end,:]) + 0.05, 2)
        diff = maxVal - minVal
        step = round(diff/4,2)
        y_ticks = np.zeros(5,dtype="float32")
        for tick in range(0,5,1):
            y_ticks[tick] = round(minVal + tick * step,2)
    # Create the figure
    fig = plt.figure(figsize=(15, 10))
    # Create a title
    plt.suptitle(title, fontsize = 25, fontweight='bold', color=colors['text'])
    # Create a time axis
    t = np.arange(0,len(data[0,:,0]),1)/fs
    # Create a channel index to plot the channels in the correct order
    # Loop through each subplot index
    for sp_index in range(1,17):
        # Create the subplot and select the index
        plt.subplot(4,4,sp_index)
        channel_index = sp_index-1
        RMS[channel_index] = np.sqrt(np.mean(data[:,:,channel_index]**2))
        plt.axhline(RMS)
        # Loop through each set
        for sets in range(0,12):
            # Plot the set data
            plt.plot(t, data[sets,:,channel_index], color=colors['signal'][sets])
            # Set axis limits
            plt.xlim(x_ticks[0], x_ticks[-1])
            plt.ylim(y_ticks[0], y_ticks[-1])
            # Set the channel number
            plt.title('Channel ' + str(channel_index + 1), fontsize = fontsize_labels, fontweight='bold')
            # Set frid
            plt.grid(True)
            # If the subplot index is on the left axis, set y-axis
            if sp_index in [1,5,9,13]:
                plt.yticks(y_ticks)
                plt.ylabel(y_label, fontsize = fontsize_labels, fontweight='bold')
                plt.rc('ytick', labelsize=fontsize_labels)
            else:
                # Unset the y axis label
                plt.yticks(y_ticks, labels = "")
            if sp_index in [13,14,15,16]:
                plt.xticks(x_ticks)
                plt.xlabel(x_label, fontsize = fontsize_labels, fontweight='bold')
                plt.rc('xtick', labelsize=fontsize_labels)
            else:
                 plt.xticks(x_ticks, labels = "")
    # Export the figure if set
    if save:
        fig.savefig('{}.eps'.format(filename),format='pdf')
        
