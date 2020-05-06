import math
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from scipy.misc import derivative
from scipy import integrate
import scipy
import sympy

#s,y = np.loadtxt("document_voids_1.txt", unpack = True)*s**2
file = np.loadtxt("CATALPTCICz0.562G960S1010008301.dat.2pcf", unpack = True)
s = file[0]
y = file[1]
plt.figure()
plt.plot(s,y, label ="model_before_fitting")
plt.legend(loc="upper right")
plt.savefig("hard_sphere_model-5.png")
plt.show()