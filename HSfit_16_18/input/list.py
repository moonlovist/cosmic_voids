import math
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from scipy.misc import derivative
from scipy import integrate
import sympy
from sympy import diff
from sympy import *
from scipy.fftpack import fft,ifft
import os
from scipy.optimize import curve_fit

#path = "/mocks"
files = os.listdir("/hpcstorage/zhaoc/share/ting/powspec_fitting/HSfit_16_18/input/mocks_ori")
files.sort()
f = open('list_mocks.dat', 'w+')
for file in files:
  f.write('%s'%file)
  f.write('\n')
f.close()

