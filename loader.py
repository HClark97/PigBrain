import tdt
import numpy as np
from scipy import signal
import scipy.fftpack 
from scipy.fft import fft, fftfreq
import matplotlib.pyplot as plt
import FunctionsP7.FFT.Power_density as pdf


#load data
data = tdt.read_block(r'C:\Users\mbj\Desktop\Uni\8. semester\Projekt\Data\Subject9-210426-124955')
fs = data.streams.RSn1.fs
data = data.streams.RSn1.data


#Plot PDF
pdf.psd_plot(data[1],fs,300)



