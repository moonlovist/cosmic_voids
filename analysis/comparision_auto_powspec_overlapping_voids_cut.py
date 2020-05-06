import numpy as np
import os
from scipy.interpolate import interp1d
from scipy.optimize import minimize
import matplotlib.pyplot as plt


#k_1, Pk_1, Variance_1 = np.loadtxt("/hpcstorage/zhaoc/share/ting/powspec_cross_1/power_spec_disjoint_voids_plus_1/powspec_dis_voids_average_1.txt", usecols=(0,1,2), unpack=True)
#k_2, Pk_2, Variance_2 = np.loadtxt("/hpcstorage/zhaoc/share/ting/powspec_cross_2/power_spec_disjoint_voids_plus_2/powspec_dis_voids_average_2.txt", usecols=(0,1,2), unpack=True)
#k_3, Pk_3, Variance_3 = np.loadtxt("/hpcstorage/zhaoc/share/ting/powspec_cross_3/power_spec_disjoint_voids_plus_3/powspec_dis_voids_average_plus_3.txt", usecols=(0,1,2), unpack=True)
k_16, Pk_16, Variance_16 = np.loadtxt("/hpcstorage/zhaoc/share/ting/powspec_cross_over/power_spec_over_voids_16/power_spec_over_voids_16/average.dat", usecols=(0,1,2), unpack=True)
k_20, Pk_20, Variance_20 = np.loadtxt("/hpcstorage/zhaoc/share/ting/powspec_cross_over/power_spec_over_voids_20/power_spec_over_voids_20/average.dat", usecols=(0,1,2), unpack=True)
k_ori, Pk_ori, Variance_ori = np.loadtxt("/hpcstorage/zhaoc/share/ting/powspec_cross_over/power_spec_over_voids_ori/power_spec_over_voids_20/average.dat", usecols=(0,1,2), unpack=True)
#Variance_1 = Variance_1/np.max(abs(Pkd))
#Pkd = Pkd/np.max(abs(Pkd))
#kb, Pb = np.loadtxt("output/HSfit_input_data_bestfit.dat", usecols=(0,1), unpack=True)
#Pb = Pb/np.max(abs(Pb))
def plot_error(x,y,error,labe,colo):
  #plt.plot(x,y, label = labe)
  plt.errorbar(x,y, yerr=error, color=colo,label = labe)
  plt.fill_between(x, y - error, y + error, color=colo, alpha=0.2)

plt.figure(figsize=(20,20))
plot_error(k_16,Pk_16,Variance_16,'auto power spectrum of overlapping voids radius cut 16 $Mpch^{-1}$','red')
plot_error(k_20,Pk_20,Variance_20,'auto power spectrum of overlapping voids radius cut 20 $Mpch^{-1}$','blue')
#plot_error(k_ori,Pk_ori,Variance_ori,'auto power spectrum of overlapping voids','yellow')
#plot_error(k_4,Pk_4,Variance_4,'powspec - 4','black')
#plot_error(k_5,Pk_5,Variance_5,'powspec - 5','green')
#plot_error(k_ori,Pk_ori,Variance_ori,'powspec - ori','magenta')
#plt.plot(kb,Pb, label = "power - spectrum - auto - correlation - model")
#plt.errorbar(kd,Pkd, yerr=Variance_1, label = "power - spectrum - voids - original")
#plt.fill_between(kd, Pkd - Variance_1, Pkd + Variance_1, color='red', alpha=0.2)
#plt.plot(k_ori, Pk_ori, label = 'powspec - original')
plt.legend(loc="lower right", fontsize=20)
plt.tick_params(labelsize=23)
plt.xscale("log")
plt.yscale("log")
plt.xlabel("$k[hMpc^{-1}]$",fontsize=30)
plt.ylabel("P(k)",fontsize=30)
#plt.xlim(0.1,1)
#plt.ylim(-20000,38000)
plt.savefig("/home/epfl/tan/first-code-document/analysis_cos/Comparison-powspec-overlapping-voids-cut")
