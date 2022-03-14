#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  5 11:17:50 2021

@author: wizard
"""
import numpy as np

def correct_baseline(data):
    """
    Perform a baseline correction by calculating average of the array and 
     subtracting the average from each point in the array

    Parameters
    ----------
    data : array
        The data array to baseline correct.

    Returns
    -------
    array
        The baseline corrected data.

    """
    # Calculate the average of the data
    baseline = np.mean(data[0:12207]) # 12207 is the index at 0.5 s (fs * 0.5 = 12207 :) )
    # Subtract the baseline from each datapoint
    return data - baseline

def correct_baseline_300ms(data, fs):
    """
    Perform a baseline correction by calculating average of the array and 
     subtracting the average from each point in the array

    Parameters
    ----------
    data : array
        The data array to baseline correct.

    Returns
    -------
    array
        The baseline corrected data.

    """
    sample_at_300ms = round(fs * 0.3)
    print(sample_at_300ms)
    # Calculate the average of the data
    baseline = np.mean(data[0:sample_at_300ms])
    # Subtract the baseline from each datapoint
    return data - baseline