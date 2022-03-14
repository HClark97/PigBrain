# -*- coding: utf-8 -*-
"""
Created on Mon Mar  7 10:17:19 2022

@author: mbj
"""

import matplotlib.pyplot as plt
import numpy as np
import tdt

data = tdt.read_block('/Users/amaliekoch/Desktop/Suzan_Acute1-210426/Subject9-210426-124955')
#print(data)

test = data.epocs 
#print(test)

test1 = data.streams.RSn1
#print(test1)

fs = data.streams.RSn1.fs
filtered_channel = np.array(data.streams.RSn1.filtered_chan)
filtered_data = np.array(data.streams.RSn1.filtered_data)

#print(data.info)

snips = data.snips
print(snips)

