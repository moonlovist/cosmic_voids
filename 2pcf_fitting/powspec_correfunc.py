#!/usr/bin/env python3
import numpy as np
import os
from scipy.interpolate import interp1d
from scipy.optimize import minimize
from hsmodel import hkfunc
#from HSfit import best_fit
from FFTlog import xicalc
from params import *
from fitfunc import *
path = "/hpcstorage/zhaoc/share/ting/powspec_fitting/HSfit_4/input/mocks_ori"
files = os.listdir(path)
files.sort()
for i in range(0,len(files)):
    name = files[i]
    file_k, file_pk, file_var = np.loadtxt("/hpcstorage/zhaoc/share/ting/powspec_fitting/HSfit_4/input/mocks_ori/%s"%name, usecols=(0,1,2), unpack = True)
    fk = interp1d(file_k, file_pk, kind='cubic')
    fhk = lambda k : fk(k)
    r, chi = xicalc(fhk, int(model_Nk), np.min(file_k), np.max(file_k), 0.01)
    np.savetxt("/hpcstorage/zhaoc/share/ting/2PCF_disjoint_voids/powspec_correlation_function_auto_disjoint_16/%s"%name, np.transpose([r,chi]), fmt = '%g')    
#kk = fk()
#plt.figure(figsize=(20,20))
#plt.plot(k_test,P_test, label = "power - spectrum - test")
#plt.plot(k_lin,P_lin, label = "power - spectrum - lin_nw")
#plt.plot(kb,Pb, label = "power - spectrum - auto - correlation - model")
#plt.errorbar(kd,Pkd, yerr=Variance_1, label = "power - spectrum - voids - original")
#plt.fill_between(kd, Pkd - Variance_1, Pkd + Variance_1, color='red', alpha=0.2)

#plt.errorbar(kd_PACHY,Pkd_PACHY, yerr=Variance_1_PACHY, label = "power - spectrum - voids - PACHY")
#plt.fill_between(kd_PACHY, Pkd_PACHY - Variance_1_PACHY, Pkd_PACHY + Variance_1_PACHY, color='red', alpha=0.2)

#plt.plot(r*32,chi, label = "correlation function 18 - 20 from powspec")

#plt.legend(loc="lower right", fontsize=20)
#plt.xscale("log")
#plt.xlim(0.01, 200)
#plt.ylim(-130000, 50000)
#plt.savefig("output/Best - fitting - 16_18 - 5")
#plt.savefig("/home/epfl/tan/first-code-document/analysis_cos/output/Best - fitting - 16_18 - 5")
