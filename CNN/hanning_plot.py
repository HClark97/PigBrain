# -*- coding: utf-8 -*-
"""
Created on Tue Apr  5 21:53:21 2022

@author: nicko
"""

import numpy as np
import matplotlib
import matplotlib.pyplot as plt

fig = plt.figure()
plt.plot(np.hanning(700))
plt.xlabel('Samples', fontweight='bold')
plt.ylabel('Amplitude', fontweight='bold')

fig.savefig('{}.eps'.format('Hanning'),format='eps')
