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
kd_4, Pkd_4, Variance_4 = np.loadtxt("/hpcstorage/zhaoc/share/ting/powspec_cross_ori/power_spec_disjoint_voids_4/average.dat", usecols=(0,1,2), unpack=True)
kd_16, Pkd_16, Variance_16 = np.loadtxt("/hpcstorage/zhaoc/share/ting/powspec_fitting/HSfit_2/input/input_data.dat", usecols=(0,1,2), unpack=True)
kd_8, Pkd_8, Variance_8 = np.loadtxt("/hpcstorage/zhaoc/share/ting/powspec_cross_ori/power_spec_disjoint_voids_8/average.dat", usecols=(0,1,2), unpack=True)
kd_12, Pkd_12, Variance_12 = np.loadtxt("/hpcstorage/zhaoc/share/ting/powspec_cross_ori/power_spec_disjoint_voids_12/average.dat", usecols=(0,1,2), unpack=True)
kd_24, Pkd_24, Variance_24 = np.loadtxt("/hpcstorage/zhaoc/share/ting/powspec_cross_ori/power_spec_disjoint_voids_24/average.dat", usecols=(0,1,2), unpack=True)
kd_28, Pkd_28, Variance_28 = np.loadtxt("/hpcstorage/zhaoc/share/ting/powspec_cross_ori/power_spec_disjoint_voids_28/average.dat", usecols=(0,1,2), unpack=True)

plt.figure(figsize=(20,20))
plot_error(kd_4,Pkd_4,Variance_4,'auto powspec of disjoint voids larger than 4 Mpc','green')
plot_error(kd_8,Pkd_8,Variance_8,'auto powspec of disjoint voids larger than 8 Mpc','blue')
plot_error(kd_20,Pkd_20,Variance_20,'auto powspec of disjoint voids larger than 20 Mpc','red')
plot_error(kd_12,Pkd_12,Variance_12,'auto powspec of disjoint voids larger than 12 Mpc','magenta')
plot_error(kd_16,Pkd_16,Variance_16,'auto powspec of disjoint voids larger than 16 Mpc','yellow')
plot_error(kd_24,Pkd_24,Variance_24,'auto powspec of disjoint voids larger than 24 Mpc','black')
#plot_error(kd_28,Pkd_28,Variance_28,'auto powspec of disjoint voids larger than 28 Mpc','cyan')
#plt.plot(kb,Pb, label = "power - spectrum - auto - correlation - model")
#plt.errorbar(kd,Pkd, yerr=Variance_1, label = "power - spectrum - voids - original")
#plt.fill_between(kd, Pkd - Variance_1, Pkd + Variance_1, color='red', alpha=0.2)
#plt.plot(k_ori, Pk_ori, label = 'powspec - original')
plt.tick_params(labelsize=23)
plt.legend(loc="lower right", fontsize=20)
plt.xscale("log")
plt.xlabel("k[hMpc-1]",fontsize=30)
plt.ylabel("P(k)",fontsize=30)
plt.xlim(0.01, 1)
plt.ylim(-320000,70000)
plt.savefig("/home/epfl/tan/first-code-document/analysis_cos/output/Analysis - cos - 10")

