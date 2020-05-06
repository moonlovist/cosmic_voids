import numpy as np
import os
from scipy.interpolate import interp1d
from scipy.optimize import minimize
import matplotlib.pyplot as plt

def plot_error(x,y,error,labe,colo):
  #plt.plot(x,y, label = labe)
  plt.errorbar(x,y, yerr=error, color=colo,label = labe)
  plt.fill_between(x, y - error, y + error, color=colo, alpha=0.2)

kd_8_10, Pkd_8_10, Variance_8_10 = np.loadtxt("/hpcstorage/zhaoc/share/ting/powspec_cross_ori/power_spec_disjoint_voids_8_10/average.dat", usecols=(0,1,2), unpack=True)
kd_10_12, Pkd_10_12, Variance_10_12 = np.loadtxt("/hpcstorage/zhaoc/share/ting/powspec_cross_ori/power_spec_disjoint_voids_10_12/average.dat", usecols=(0,1,2), unpack=True)
kd_12_14, Pkd_12_14, Variance_12_14 = np.loadtxt("/hpcstorage/zhaoc/share/ting/powspec_cross_ori/power_spec_disjoint_voids_12_14/average.dat", usecols=(0,1,2), unpack=True)
kd_14_16, Pkd_14_16, Variance_14_16 = np.loadtxt("/hpcstorage/zhaoc/share/ting/powspec_cross_ori/power_spec_disjoint_voids_14_16/average.dat", usecols=(0,1,2), unpack=True)
kd_16_18, Pkd_16_18, Variance_16_18 = np.loadtxt("/hpcstorage/zhaoc/share/ting/powspec_cross_ori/power_spec_disjoint_voids_16_18/average.dat", usecols=(0,1,2), unpack=True)
kd_18_20, Pkd_18_20, Variance_18_20 = np.loadtxt("/hpcstorage/zhaoc/share/ting/powspec_cross_ori/power_spec_disjoint_voids_18_20/average.dat", usecols=(0,1,2), unpack=True)
kd_20_22, Pkd_20_22, Variance_20_22 = np.loadtxt("/hpcstorage/zhaoc/share/ting/powspec_cross_ori/power_spec_disjoint_voids_20_22/average.dat", usecols=(0,1,2), unpack=True)
kd_22_24, Pkd_22_24, Variance_22_24 = np.loadtxt("/hpcstorage/zhaoc/share/ting/powspec_cross_ori/power_spec_disjoint_voids_22_24/average.dat", usecols=(0,1,2), unpack=True)

plt.figure(figsize=(20,20))
plot_error(kd_8_10,Pkd_8_10,Variance_8_10,'auto powspec of disjoint voids between 8 and 10 Mpc','green')
plot_error(kd_10_12,Pkd_10_12,Variance_10_12,'auto powspec of disjoint voids between 10 and 12 Mpc','blue')
plot_error(kd_12_14,Pkd_12_14,Variance_12_14,'auto powspec of disjoint voids between 12 and 14 Mpc','red')
plot_error(kd_14_16,Pkd_14_16,Variance_14_16,'auto powspec of disjoint voids between 14 and 16 Mpc','magenta')
plot_error(kd_16_18,Pkd_16_18,Variance_16_18,'auto powspec of disjoint voids between 16 and 18 Mpc','yellow')
plot_error(kd_18_20,Pkd_18_20,Variance_18_20,'auto powspec of disjoint voids between 18 and 20 Mpc','black')
plot_error(kd_20_22,Pkd_20_22,Variance_20_22,'auto powspec of disjoint voids between 20 and 22 Mpc','cyan')
plot_error(kd_22_24,Pkd_22_24,Variance_22_24,'auto powspec of disjoint voids between 22 and 24 Mpc','white')

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
plt.ylim(-220000,70000)
plt.savefig("/home/epfl/tan/first-code-document/analysis_cos/output/Analysis - cos - 11")

