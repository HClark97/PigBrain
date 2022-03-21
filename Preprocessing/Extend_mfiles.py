# -*- coding: utf-8 -*-
"""
Created on Mon Mar 21 09:59:11 2022

@author: mbj
"""
import loader as l
from scipy.io import savemat
#LOAD DATA
data1=l.load_mat_file(1) #<-- til at loade epochs fra subject 1
# data2=l.load_mat_file(2) #<-- til at loade epochs fra subject 9
# data3=l.load_mat_file(3) #<-- til at loade epochs fra subject 1 without RS4

# #Adds data1, data2 and data3 together and saves as .mat file
# data1['NoxERP']['block'].extend(data2['NoxERP']['block'])
# data1['NonnoxERP']['block'].extend(data2['NonnoxERP']['block'])
# data1['NoxERP']['block'].extend(data3['NoxERP']['block'])
# data1['NonnoxERP']['block'].extend(data3['NonnoxERP']['block'])
# savemat("nonpreprocessed_data.mat", data1)

# fs = 6103.5156 # sample frequency