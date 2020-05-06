from scipy.optimize import minimize
import matplotlib.pyplot as plt
import numpy as np
def plot_error(x,y,error,labe,colo):
  #plt.plot(x,y, label = labe)
  plt.errorbar(x,y, yerr=error, color=colo,label = labe)
  plt.fill_between(x, y - error, y + error, color=colo, alpha=0.2)

#kd_20, Pkd_20, Variance_20 = np.loadtxt("/hpcstorage/zhaoc/share/ting/2PCF_disjoint_voids/powspec_correlation_function_auto_disjoint_20/work_average.dat", usecols=(0,1,2), unpack=True)
kd_16_18, Pkd_16_18, Variance_16_18 = np.loadtxt("/hpcstorage/zhaoc/share/ting/2PCF_disjoint_voids/powspec_correlation_function_auto_disjoint_16_18/work_average.dat", usecols=(0,1,2), unpack=True)
kd_18_20, Pkd_18_20, Variance_18_20 = np.loadtxt("/hpcstorage/zhaoc/share/ting/2PCF_disjoint_voids/powspec_correlation_function_auto_disjoint_18_20/work_average.dat", usecols=(0,1,2), unpack=True)
kd_16_20, Pkd_16_20, Variance_16_20 = np.loadtxt("/hpcstorage/zhaoc/share/ting/2PCF_disjoint_voids/powspec_correlation_function_auto_disjoint_16_20/work_average.dat", usecols=(0,1,2), unpack=True)
kd_16, Pkd_16, Variance_16 = np.loadtxt("/hpcstorage/zhaoc/share/ting/2PCF_disjoint_voids/powspec_correlation_function_auto_disjoint_16/work_average.dat", usecols=(0,1,2), unpack=True)

plt.figure(figsize=(20,20))
#plot_error(40*kd_20,Pkd_20,Variance_20,'from powspec auto correlation function of disjoint voids 20','red')
plot_error(40*kd_16_20,Pkd_16_20,Variance_16_20,'from powspec auto correlation function of disjoint voids 16 - 20','magenta')
plot_error(40*kd_18_20,Pkd_18_20,Variance_18_20,'from powspec auto correlation function of disjoint voids 18 - 20','green')
plot_error(40*kd_16_18,Pkd_16_18,Variance_16_18,'from powspec auto correlation function of disjoint voids 16 - 18','blue')
plot_error(40*kd_16,Pkd_16,Variance_16,'from powspec auto correlation function of disjoint voids 16','yellow')
#plt.plot(kb,Pb, label = "power - spectrum - auto - correlation - model")
#plt.errorbar(kd,Pkd, yerr=Variance_1, label = "power - spectrum - voids - original")
#plt.fill_between(kd, Pkd - Variance_1, Pkd + Variance_1, color='red', alpha=0.2)
#plt.plot(k_ori, Pk_ori, label = 'powspec - original')
plt.legend(loc="lower right", fontsize=20)
#plt.xscale("log")
plt.xlim(0.01, 12.5)
#plt.ylim(-200000,50000)
plt.savefig("/home/epfl/tan/first-code-document/analysis_cos/output/correlation_function_from_powspec_auto_disjoint_voids - 1")
