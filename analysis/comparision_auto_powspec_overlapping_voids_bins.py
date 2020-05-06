import numpy as np
import os
from scipy.interpolate import interp1d
from scipy.optimize import minimize
import matplotlib.pyplot as plt


k_16_20, Pk_16_20, Variance_16_20 = np.loadtxt("/hpcstorage/zhaoc/share/ting/powspec_cross_over/power_spec_over_voids_16_20/power_spec_over_voids_16_20/average.dat", usecols=(0,1,2), unpack=True)
k_22_24, Pk_22_24, Variance_22_24 = np.loadtxt("/hpcstorage/zhaoc/share/ting/powspec_cross_over/power_spec_over_voids_22_24/power_spec_over_voids_22_24/average.dat", usecols=(0,1,2), unpack=True)
k_20_22, Pk_20_22, Variance_20_22 = np.loadtxt("/hpcstorage/zhaoc/share/ting/powspec_cross_over/power_spec_over_voids_20_22/power_spec_over_voids_20_22/average.dat", usecols=(0,1,2), unpack=True)
k_16_18, Pk_16_18, Variance_16_18 = np.loadtxt("/hpcstorage/zhaoc/share/ting/powspec_cross_over/power_spec_over_voids_16_18/power_spec_over_voids_16_18/average.dat", usecols=(0,1,2), unpack=True)
k_18_20, Pk_18_20, Variance_18_20 = np.loadtxt("/hpcstorage/zhaoc/share/ting/powspec_cross_over/power_spec_over_voids_18_20/power_spec_over_voids_18_20/average.dat", usecols=(0,1,2), unpack=True)
#k_ori, Pk_ori, Variance_ori = np.loadtxt("/hpcstorage/zhaoc/share/ting/powspec_cross_over/power_spec_over_voids_ori/power_spec_over_voids_20/average.dat", usecols=(0,1,2), unpack=True)
#Variance_1 = Variance_1/np.max(abs(Pkd))
#Pkd = Pkd/np.max(abs(Pkd))
kb, Pb = np.loadtxt("Pnw_1.dat", usecols=(0,1), unpack=True)
#Pb = Pb/np.max(abs(Pb))
def plot_error(x,y,error,labe,colo):
  #plt.plot(x,y, label = labe)
  plt.errorbar(x,y, yerr=error, color=colo,label = labe)
  plt.fill_between(x, y - error, y + error, color=colo, alpha=0.2)
func_1 = interp1d(kb, Pb, kind='cubic')
Pb = func_1(k_16_20)
plt.figure(figsize=(20,20))
plot_error(k_16_18,Pk_16_18/Pb,Variance_16_18/Pb,'auto power spectrum of overlapping voids radius bins 16 - 18 $Mpch^{-1}$','red')
plot_error(k_18_20,Pk_18_20/Pb,Variance_18_20/Pb,'auto power spectrum of overlapping voids radius bins 18 - 20 $Mpch^{-1}$','blue')
plot_error(k_20_22,Pk_20_22/Pb,Variance_20_22/Pb,'auto power spectrum of overlapping voids radius bins 20 - 22 $Mpch^{-1}$','yellow')
plot_error(k_22_24,Pk_22_24/Pb,Variance_22_24/Pb,'auto power spectrum of overlapping voids radius bins 22 - 24 $Mpch^{-1}$','black')
plot_error(k_16_20,Pk_16_20/Pb,Variance_16_20/Pb,'auto power spectrum of overlapping voids radius bins 16 - 20 $Mpch^{-1}$','green')
#plot_error(k_ori,Pk_ori,Variance_ori,'powspec - ori','magenta')
#plt.plot(kb,Pb, label = "power - spectrum - auto - correlation - model")
#plt.errorbar(kd,Pkd, yerr=Variance_1, label = "power - spectrum - voids - original")
#plt.fill_between(kd, Pkd - Variance_1, Pkd + Variance_1, color='red', alpha=0.2)
#plt.plot(k_ori, Pk_ori, label = 'powspec - original')
plt.legend(loc="lower left", fontsize=20)
plt.tick_params(labelsize=23)
plt.xscale("log")
plt.yscale("log")
plt.xlabel("$k[hMpc^{-1}]$",fontsize=30)
plt.ylabel("P(k)",fontsize=30)
#plt.xlim(0.1,1)
#plt.ylim(-20000,38000)
plt.savefig("/home/epfl/tan/first-code-document/analysis_cos/Comparison-powspec-overlapping-voids-bins-2")
