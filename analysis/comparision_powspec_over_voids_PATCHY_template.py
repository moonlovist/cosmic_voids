import numpy as np
import os
from scipy.interpolate import interp1d
from scipy.optimize import minimize
import matplotlib.pyplot as plt

def plot_error(x,y,error,labe,colo):
  #plt.plot(x,y, label = labe)
  plt.errorbar(x,y, yerr=error, color=colo,label = labe)
  plt.fill_between(x, y - error, y + error, color=colo, alpha=0.2)

kd_PATCHY, Pkd_PATCHY, Variance_PATCHY = np.loadtxt("/hpcstorage/zhaoc/share/ting/PATCHY_pre_halos/powspec_PATCHY_halos/catelogues/average.dat", usecols=(0,1,2), unpack=True)
kd_template, Pkd_template, Variance_template = np.loadtxt("/hpcstorage/zhaoc/share/ting/powspec_cross_over/power_spec_over_voids_ori/power_spec_over_voids_20/average.dat", usecols=(0,1,2), unpack=True)
num = Pkd_template[0]/Pkd_PATCHY[0]
fig = plt.figure(figsize=(20,20))
#plt.plot(kd_1,Pkd_1,label = "cosmology - 1",color = "green")
#plt.plot(kd_2,Pkd_2,label = "cosmology - 2",color = "blue")
#plt.plot(kb,Pb, label = "power - spectrum - auto - correlation - model")
plt.errorbar(kd_PATCHY,Pkd_PATCHY, yerr=Variance_PATCHY, label = "power spectrum of overlapping voids of PATCHY mocks", color = 'red')
plt.fill_between(kd_PATCHY, Pkd_PATCHY - Variance_PATCHY, Pkd_PATCHY + Variance_PATCHY, color='red', alpha=0.2)
plt.errorbar(kd_template,Pkd_template/num, yerr=Variance_template/num, label = "power spectrum of overlapping voids of template mocks",color = 'blue')
plt.fill_between(kd_template, Pkd_template/num - Variance_template/num, Pkd_template/num + Variance_template/num, color='blue', alpha=0.2)
#plt.plot(k_ori, Pk_ori, label = 'powspec - original')
plt.tick_params(labelsize=23)
plt.legend(loc="upper right", fontsize=20)
plt.yscale("log")
plt.xscale("log")
plt.xlim(0.01, 1)
plt.xlabel("$k[hMpc^{-1}]$",fontsize=30)
plt.ylabel("P(k)",fontsize=30)
plt.ylim(100,70000)



#ax1 = fig.add_axes([0.1,0.001,0.5,0.5])
#ax1.plot(kd_1,Pkd_1,label = "cosmology - 1",color = "green")
#ax1.plot(kd_2,Pkd_2,label = "cosmology - 2",color = "blue")
#ax1.plot(kd_3,Pkd_3,label = "cosmology - 3",color = "red")
#ax1.plot(kd_4,Pkd_4,label = "cosmology - 4",color = "magenta")
#ax1.plot(kd_5,Pkd_5,label = "cosmology - 5",color = "yellow")
#ax1.yscale("log")
#ax1.xscale("log")


plt.savefig("/home/epfl/tan/first-code-document/analysis_cos/output/Comparison of power spectrum of overlapping voids of PATCHY and Template")

