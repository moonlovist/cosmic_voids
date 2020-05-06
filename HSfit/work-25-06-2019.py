import numpy as np
import os
from scipy.interpolate import interp1d
from scipy.optimize import minimize
from hsmodel import hkfunc
import matplotlib.pyplot as plt


kd, Pkd, Variance_1 = np.loadtxt("input/input_data.dat", usecols=(0,1,2), unpack=True)

#Variance_1 = Variance_1/np.max(abs(Pkd))
#Pkd = Pkd/np.max(abs(Pkd))
kb, Pb = np.loadtxt("output/HSfit_input_data_bestfit.dat", usecols=(0,1), unpack=True)
#Pb = Pb/np.max(abs(Pb))

plt.figure(figsize=(20,20))
plt.plot(kb,Pb, label = "power - spectrum - auto - correlation - model")
plt.errorbar(kd,Pkd, yerr=Variance_1, label = "power - spectrum - voids - original")
plt.fill_between(kd, Pkd - Variance_1, Pkd + Variance_1, color='red', alpha=0.2)
plt.legend(loc="lower right", fontsize=20)
plt.xscale("log")
plt.xlim(0.01, 0.7)
plt.ylim(-100000, 50000)
plt.savefig("output/Best - fitting - 1")
