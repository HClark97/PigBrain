# -*- coding: utf-8 -*-
"""
Created on Tue Mar 22 11:26:24 2022

@author: nicko
"""

import torch as t
import loader as l
import matplotlib.pyplot as plt

#data=l.load_mat_file()
stim=data['NonnoxERP']['block'][10]['channel'][0]['ERPs'][:,0]
plt.plot(stim)

t.stft(stim)

