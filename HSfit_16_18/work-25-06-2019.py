#!/usr/bin/env python3
import numpy as np
import os
from scipy.interpolate import interp1d
from scipy.optimize import minimize
from hsmodel import hkfunc
import matplotlib.pyplot as plt
#from HSfit import best_fit
from FFTlog import xicalc
from params import *
from fitfunc import *

r_ori, chi_ori, chi_var = np.loadtxt("/hpcstorage/zhaoc/share/ting/2PCF_disjoint_voids/correlation_function_auto_disjoint_16_18/average.dat", usecols=(0,1,2), unpack=True)

kd, Pkd, Variance_1 = np.loadtxt("input/input_data.dat", usecols=(0,1,2), unpack=True)
k_lin, P_lin = np.loadtxt("input/Pnw.dat", usecols=(0,1), unpack=True)
#Variance_1 = Variance_1/np.max(abs(Pkd))
#Pkd = Pkd/np.max(abs(Pkd))
kb, Pb = np.loadtxt("output/HSfit_input_data_bestfit.dat", usecols=(0,1), unpack=True)
#Pb = Pb/np.max(abs(Pb))

kd_PACHY, Pkd_PACHY, Variance_1_PACHY = np.loadtxt("/hpcstorage/zhaoc/share/ting/PATCHY_pre_voids/powspec_PACHY_dis_voids_16_18/catelogues/average.dat", usecols=(0,1,2), unpack=True)
test = np.loadtxt("/hpcstorage/zhaoc/share/ting/PATCHY_pre_voids/powspec_PACHY_dis_voids_16_18/catelogues/average.dat", usecols=(0,1,2), unpack=True)

#result = test[test[0]<0.7]

k_2, P_2 = hkfunc([33.513739,0.40260519,37490.99], model_kmin, model_kmax, model_Nk)
fk = interp1d(k_2, P_2, kind='cubic')

#for i in range(len(k_test)):
#  if k_test[i]>0.6:
#    P_test[i]=0
#fk = interp1d(k_test, P_test, kind='cubic')
#k_test = k_test[0.00001<k_test]
#P_test = fk(k_test)

#P_test = fk(kd)
#k_test = kd
fhk = lambda k : fk(k)
#print(np.min(k_test),np.max(k_test))
r, chi = xicalc(fhk, int(model_Nk), np.min(k_2), np.max(k_2), 0.01)

fk_1 = interp1d(kd,Pkd, kind='cubic')
fhk_1 = lambda k : fk_1(k)
#print(np.min(k_test),np.max(k_test))
r_1, chi_1 = xicalc(fhk_1, int(model_Nk), np.min(kd), np.max(kd), 0.01)

fk_2 = interp1d(kd_PACHY,Pkd_PACHY, kind='cubic')
fhk_2 = lambda k : fk_2(k)
#print(np.min(k_test),np.max(k_test))
r_2, chi_2 = xicalc(fhk_2, int(model_Nk), np.min(kd_PACHY), np.max(kd_PACHY), 0.01)

x = np.linspace(0.5,195.5,100)
#kk = fk()
plt.figure(figsize=(20,20))
plt.plot(k_2,P_2, label = "fitting from Hard sphere model - 2", color='blue')
#plt.plot(k_lin,P_lin, label = "power - spectrum - lin_nw")
plt.plot(kb,Pb, label = "fitting from Hard sphere model - 1", color='black')
plt.errorbar(kd,Pkd, yerr=Variance_1, label = "power spectrum of voids from Template of radius bins 16 - 18 Mpc", color='red')
plt.fill_between(kd, Pkd - Variance_1, Pkd + Variance_1, color='red', alpha=0.2)

plt.errorbar(kd_PACHY,Pkd_PACHY, yerr=Variance_1_PACHY, label = "power spectrum of voids from PATCHY of radius bins 16 - 18 Mpc", color='yellow')
plt.fill_between(kd_PACHY, Pkd_PACHY - Variance_1_PACHY, Pkd_PACHY + Variance_1_PACHY, color='yellow', alpha=0.2)
#plt.plot(r_ori,chi_ori, label = "correlation function 18 - 20 ")

#plt.plot(r,chi, label = "correlation function 18 - 20 from powspec - model")
#plt.plot(r_1*32,chi_1, label = "correlation function 18 - 20 from powspec - template")
#plt.plot(r_2*32,chi_2, label = "correlation function 18 - 20 from powspec - PACHY")
#plt.legend(loc="lower right", fontsize=20)
#plt.xscale("log")
#plt.xlim(5, 200)

plt.legend(loc="lower right", fontsize=20)
plt.tick_params(labelsize=23)
plt.xlabel("k[hMpc-1]",fontsize=30)
plt.ylabel("P(k)",fontsize=30)
plt.xscale("log")
plt.xlim(0.01, 0.7)
#plt.ylim(-1.5, 1)
plt.ylim(-110000, 30000)
#plt.savefig("output/Best - fitting - 16_18 - 3")
plt.savefig("/home/epfl/tan/first-code-document/analysis_cos/output/Best - fitting - 16_18 - 3")
