import math
import numpy as np
from scipy.interpolate import interp1d
from scipy.misc import derivative
from scipy import integrate
import sympy
from sympy import diff
from sympy import *
from scipy.fftpack import fft,ifft
import os
from scipy.optimize import curve_fit

file = np.loadtxt("/hpcstorage/zhaoc/share/ting/2PCF_disjoint_voids/powspec_correlation_function_auto_disjoint_20/power_spec_ori_dis_voids_100.dat",unpack = True)

path = "/hpcstorage/zhaoc/share/ting/2PCF_disjoint_voids/powspec_correlation_function_auto_disjoint_20"
files = os.listdir(path)
files.sort()
x = file[0]
y = file[1]
variance = np.arange(len(file[0]))
for i in range(len(variance)):
  variance[i] = 0
  y[i] = 0

"""get the average value"""
for i in range(0,100):
    name = files[i]
    file = np.loadtxt("/hpcstorage/zhaoc/share/ting/2PCF_disjoint_voids/powspec_correlation_function_auto_disjoint_20/%s"%name,unpack = True)
    y = y + file[1]
y = y/100

"""get the variance"""
for i in range(0,100):
    name = files[i]
    file = np.loadtxt("/hpcstorage/zhaoc/share/ting/2PCF_disjoint_voids/powspec_correlation_function_auto_disjoint_20/%s"%name,unpack = True)
    variance = variance + (y - file[1])**2


variance = np.sqrt(variance/100)
np.savetxt("work_average.dat", np.transpose([x,y,variance]), fmt = '%g')
#np.savez("disjoint_void_average.npz",x,y,variance)
#df.to_csv("disjoint_void_average.csv")
#np.savetxt("disjoint_void_average.txt",x,y,variance)
#print(file)
