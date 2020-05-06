import numpy as np
import numpy.fft as fft
#import scipy.special as sp
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import pandas as pd
import os


C = np.loadtxt("/home/epfl/tan/first-code-document/disjoint_voids_pow_spectrum_2/disjoint_R16/work - covariance matrice - power spectrum - 4.txt",unpack = True)

correlation_data = np.loadtxt("work_disjoint_void_power_spectrum_average_2.txt",unpack = True)
print(np.max(correlation_data))
correlation_data = correlation_data/np.max(correlation_data)
correlation_model = np.loadtxt("fitting_power_spectrum_disjoint_void_average_2.txt",unpack = True)
correlation_model = correlation_model[1]
#correlation_model_function = interp1d(correlation_model[0], correlation_model[1], kind='cubic')
#correlation_model = correlation_model_function(correlation_data[0])

C_1 = np.linalg.inv(C)
C_1 = C_1*(100-489-2)/99.0
#print(correlation_data)
#print(correlation_model)
#print(np.transpose(correlation_data[1] - correlation_model))
print(C)
print(C_1)
#print( np.dot(np.transpose(correlation_data[1] - correlation_model),C_1))
chi_2 = np.dot(np.dot(np.transpose(correlation_data[1] - correlation_model),C_1),(correlation_data[1] - correlation_model))
print(chi_2)

