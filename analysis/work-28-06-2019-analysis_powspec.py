import numpy as np
import os
from scipy.interpolate import interp1d
from scipy.optimize import minimize
import matplotlib.pyplot as plt


k_1, Pk_1, Variance_1 = np.loadtxt("/hpcstorage/zhaoc/share/ting/powspec_cross_1/power_spec_disjoint_voids_plus_1/powspec_dis_voids_average_1.txt", usecols=(0,1,2), unpack=True)
k_2, Pk_2, Variance_2 = np.loadtxt("/hpcstorage/zhaoc/share/ting/powspec_cross_2/power_spec_disjoint_voids_plus_2/powspec_dis_voids_average_2.txt", usecols=(0,1,2), unpack=True)
k_3, Pk_3, Variance_3 = np.loadtxt("/hpcstorage/zhaoc/share/ting/powspec_cross_3/power_spec_disjoint_voids_plus_3/powspec_dis_voids_average_plus_3.txt", usecols=(0,1,2), unpack=True)
k_4, Pk_4, Variance_4 = np.loadtxt("/hpcstorage/zhaoc/share/ting/powspec_cross_4/power_spec_disjoint_voids_plus_4/powspec_dis_voids_average_4.txt", usecols=(0,1,2), unpack=True)
k_5, Pk_5, Variance_5 = np.loadtxt("/hpcstorage/zhaoc/share/ting/powspec_cross_5/power_spec_disjoint_voids_plus_5/powspec_dis_voids_average_5.txt", usecols=(0,1,2), unpack=True)
k_ori, Pk_ori, Variance_ori = np.loadtxt("/hpcstorage/zhaoc/share/ting/powspec_cross_ori/power_spec_disjoint_voids_ori/powspec_dis_voids_average_ori.txt", usecols=(0,1,2), unpack=True)
#Variance_1 = Variance_1/np.max(abs(Pkd))
#Pkd = Pkd/np.max(abs(Pkd))
#kb, Pb = np.loadtxt("output/HSfit_input_data_bestfit.dat", usecols=(0,1), unpack=True)
#Pb = Pb/np.max(abs(Pb))
def plot_error(x,y,error,labe,colo):
  #plt.plot(x,y, label = labe)
  plt.errorbar(x,y, yerr=error, color=colo,label = labe)
  plt.fill_between(x, y - error, y + error, color=colo, alpha=0.2)

plt.figure(figsize=(20,20))
plot_error(k_1,Pk_1,Variance_1,'powspec - 1','red')
plot_error(k_2,Pk_2,Variance_2,'powspec - 2','blue')
plot_error(k_3,Pk_3,Variance_3,'powspec - 3','yellow')
plot_error(k_4,Pk_4,Variance_4,'powspec - 4','black')
plot_error(k_5,Pk_5,Variance_5,'powspec - 5','green')
#plot_error(k_ori,Pk_ori,Variance_ori,'powspec - ori','magenta')
#plt.plot(kb,Pb, label = "power - spectrum - auto - correlation - model")
#plt.errorbar(kd,Pkd, yerr=Variance_1, label = "power - spectrum - voids - original")
#plt.fill_between(kd, Pkd - Variance_1, Pkd + Variance_1, color='red', alpha=0.2)
#plt.plot(k_ori, Pk_ori, label = 'powspec - original')
plt.legend(loc="lower right", fontsize=20)
plt.tick_params(labelsize=23)
plt.xscale("log")
plt.xlabel("$k[hMpc^{-1}]$",fontsize=30)
plt.ylabel("P(k)",fontsize=30)
plt.xlim(0.1,1)
plt.ylim(-20000,38000)
plt.savefig("/home/epfl/tan/first-code-document/analysis_cos/Analysis - cos - 4")
