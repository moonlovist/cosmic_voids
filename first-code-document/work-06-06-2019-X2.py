import numpy as np
import numpy.fft as fft
import scipy.special as sp
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import pandas as pd
import os


C = np.loadtxt("/home/epfl/tan/first-code-document/16R/covariance matrice.txt",unpack = True)

correlation_data = np.loadtxt("disjoint_void_average.txt",unpack = True)

correlation_model = np.loadtxt("fitting_correlation_function_disjoint_void_average_3.txt",unpack = True)
correlation_model_function = interp1d(correlation_model[0], correlation_model[1], kind='cubic')
correlation_model = correlation_model_function(correlation_data[0])

C_1 = np.linalg.inv(C)
C_1 = C_1*(100-50-2)/99.0
chi_2 = np.dot(np.dot(np.transpose(correlation_data[1] - correlation_model),C_1),(correlation_data[1] - correlation_model))
print(chi_2)

