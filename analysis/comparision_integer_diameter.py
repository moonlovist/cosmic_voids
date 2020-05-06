import numpy as np
import os
from scipy.interpolate import interp1d
from scipy.optimize import minimize
import matplotlib.pyplot as plt

def plot_error(x,y,error,labe,colo):
  #plt.plot(x,y, label = labe)
  plt.errorbar(x,y, yerr=error, color=colo,label = labe)
  plt.fill_between(x, y - error, y + error, color=colo, alpha=0.2)

kd_20, Pkd_20, Variance_20 = np.loadtxt("/hpcstorage/zhaoc/share/ting/powspec_fitting/HSfit_20/input/input_data.dat", usecols=(0,1,2), unpack=True)
kd_16_20, Pkd_16_20, Variance_16_20 = np.loadtxt("/hpcstorage/zhaoc/share/ting/powspec_fitting/HSfit_16_20/input/input_data.dat", usecols=(0,1,2), unpack=True)
kd_16, Pkd_16, Variance_16 = np.loadtxt("/hpcstorage/zhaoc/share/ting/powspec_fitting/HSfit_2/input/input_data.dat", usecols=(0,1,2), unpack=True)
kd_18_20, Pkd_18_20, Variance_18_20 = np.loadtxt("/hpcstorage/zhaoc/share/ting/powspec_fitting/HSfit_18_20/input/input_data.dat", usecols=(0,1,2), unpack=True)
kd_16_18, Pkd_16_18, Variance_16_18 = np.loadtxt("/hpcstorage/zhaoc/share/ting/powspec_fitting/HSfit_16_18/input/input_data.dat", usecols=(0,1,2), unpack=True)


plt.figure(figsize=(20,20))
plot_error(kd_18_20,Pkd_18_20,Variance_18_20,'powspec - 18_20','green')
plot_error(kd_16_18,Pkd_16_18,Variance_16_18,'powspec - 16_18','blue')
plot_error(kd_20,Pkd_20,Variance_20,'powspec - 20','red')
plot_error(kd_16_20,Pkd_16_20,Variance_16_20,'powspec - 16_20','magenta')
plot_error(kd_16,Pkd_16,Variance_16,'powspec - 16','yellow')
#plt.plot(kb,Pb, label = "power - spectrum - auto - correlation - model")
#plt.errorbar(kd,Pkd, yerr=Variance_1, label = "power - spectrum - voids - original")
#plt.fill_between(kd, Pkd - Variance_1, Pkd + Variance_1, color='red', alpha=0.2)
#plt.plot(k_ori, Pk_ori, label = 'powspec - original')
plt.legend(loc="lower right", fontsize=20)
plt.xscale("log")
plt.xlim(0.001, 1)
plt.ylim(-200000,50000)
plt.savefig("/home/epfl/tan/first-code-document/analysis_cos/output/Analysis - cos - 9")

