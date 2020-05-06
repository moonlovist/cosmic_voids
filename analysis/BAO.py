import numpy as np
import os
from scipy.interpolate import interp1d
from scipy.optimize import minimize
import matplotlib.pyplot as plt

def plot_error(x,y,error,labe,colo):
  #plt.plot(x,y, label = labe)
  plt.errorbar(x,y, yerr=error, color=colo,label = labe)
  plt.fill_between(x, y - error, y + error, color=colo, alpha=0.2)

kd_1, Pkd_1 = np.loadtxt("Pnw_1.dat", usecols=(0,1), unpack=True)
#kd_2, Pkd_2 = np.loadtxt("Pnw_2.dat", usecols=(0,1), unpack=True)
#kd_3, Pkd_3 = np.loadtxt("Pnw_3.dat", usecols=(0,1), unpack=True)
#kd_4, Pkd_4 = np.loadtxt("Pnw_4.dat", usecols=(0,1), unpack=True)
#kd_5, Pkd_5 = np.loadtxt("Pnw_5.dat", usecols=(0,1), unpack=True)
kd_PATCHY, Pkd_PATCHY, Variance_PATCHY = np.loadtxt("/hpcstorage/zhaoc/share/ting/powspec_cross_over/power_spec_over_voids_ori/power_spec_over_voids_20/average.dat", usecols=(0,1,2), unpack=True)
#m = Pkd_PATCHY[10]/Pkd_1[10]
fig = plt.figure(figsize=(20,20))
plt.plot(kd_1,Pkd_1,label = "linear power spectrum of dark matter density field",color = "red")
#plot_error(kd_PATCHY,Pkd_PATCHY, Variance_PATCHY, label = "power spectrum of overlepping voids larger than 16 $Mpch^{-1}$",color = "blue")
plot_error(kd_PATCHY,Pkd_PATCHY/100, Variance_PATCHY/100,"power spectrum of overlepping voids larger than 16 $Mpch^{-1}$","blue")
#plt.plot(kd_2,Pkd_2,label = "cosmology - 2",color = "blue")
#plt.plot(kd_3,Pkd_3,label = "cosmology - 3",color = "red")
#plt.plot(kd_4,Pkd_4,label = "cosmology - 4",color = "magenta")
#plt.plot(kd_5,Pkd_5,label = "cosmology - 5",color = "yellow")
#plt.plot(kb,Pb, label = "power - spectrum - auto - correlation - model")
#plt.errorbar(kd,Pkd, yerr=Variance_1, label = "power - spectrum - voids - original")
#plt.fill_between(kd, Pkd - Variance_1, Pkd + Variance_1, color='red', alpha=0.2)
#plt.plot(k_ori, Pk_ori, label = 'powspec - original')
plt.legend(loc="upper right", fontsize=20)
plt.yscale("log")
plt.xscale("log")
plt.xlim(0.01, 1)
plt.ylim(50,50000)


#ax1 = fig.add_axes([0.1,0.001,0.5,0.5])
#ax1.plot(kd_1,Pkd_1,label = "cosmology - 1",color = "green")
#ax1.plot(kd_2,Pkd_2,label = "cosmology - 2",color = "blue")
#ax1.plot(kd_3,Pkd_3,label = "cosmology - 3",color = "red")
#ax1.plot(kd_4,Pkd_4,label = "cosmology - 4",color = "magenta")
#ax1.plot(kd_5,Pkd_5,label = "cosmology - 5",color = "yellow")
#ax1.yscale("log")
#ax1.xscale("log")


plt.savefig("/home/epfl/tan/first-code-document/analysis_cos/output/BAO-1")

