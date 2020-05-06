#!/usr/bin/env python3
import numpy as np
import os
from scipy.interpolate import interp1d
from scipy.optimize import minimize
from params import *
from fitfunc import *
from hsmodel import hkfunc
import matplotlib.pyplot as plt

if not os.path.isdir(output_dir):
  raise FileNotFoundError('output_dir "{}" does not exist.'.format(output_dir))

bname = input_data.split('/')[-1]
bname = '.'.join(bname.split('.')[:-1])
parfile = '{}/HSfit_{}_param.dat'.format(output_dir, bname)
bestfile = '{}/HSfit_{}_bestfit.dat'.format(output_dir, bname)

print('Read the input P(k) to be fitted.')
kd, Pkd, Variance_1 = np.loadtxt(input_data, usecols=(0,1,2), unpack=True)
kd = kd[8:47]
Pkd = Pkd[8:47]
print('Read/Compute the covariance matrix.')
Nmock, icov = get_icov(input_mocks, cov_file, compute_cov, save_cov)

print('Retrieve the fitting range.')
kf, Pf, icovf = get_range(kd, Pkd, fitmin, fitmax, Nmock, icov)


def chi2_func(params):
  d, rho, A = params[0], params[1], params[2]
  if A < 0 or rho > 6.0 / np.pi or \
    model_kmin / d > kf[0] or model_kmax / d < kf[-1]:
    return 1e199

  km, Pm = hkfunc(params, model_kmin, model_kmax, model_Nk)
  if km[0] > kf[0] or km[-1] < kf[-1]:
    raise Exception('k range for model is smaller than that of data.')

  Pint = interp1d(km, Pm, kind='cubic')
  Pmodel = Pint(kf)

  diff = Pmodel - Pf
  chi2 = np.dot(diff.T, np.dot(icovf, diff))
  return chi2

def best_fit(params):
  return hkfunc(params, model_kmin, model_kmax, model_Nk)

print('Start fitting.')
npar = 3
ic = np.ones(npar)
ic = [34.5, 0.35, 48000]
res = minimize(chi2_func, ic, method='Nelder-Mead')

bestpar = res.x
#bestpar = [50.00041, 0.80821764, 3.4387081e-11]
chi2 = chi2_func(bestpar)

with open(parfile, 'w') as f:
  f.write('# d rho min_chi2\n')
  f.write('{:.8g} {:.8g} {:.8g} {:g}\n'.format(res.x[0], res.x[1], res.x[2], \
        chi2))

kb, Pb = best_fit(bestpar)
np.savetxt(bestfile, np.transpose([kb,Pb]), fmt='%.8g')


plt.figure(figsize=(20,20))
plt.plot(kb,Pb, label = "power - spectrum - auto - correlation - model")
plt.errorbar(kd,Pkd, yerr=Variance_1, label = "power - spectrum - voids - original")
plt.fill_between(kd, Pkd - Variance_1, Pkd + Variance_1, color='red', alpha=0.2)
plt.legend(loc="lower right", fontsize=20)
#plt.xlim(0.00001, 0.0001)
#plt.ylim(-0.01, 0.01)
#plt.xscale("log")
#plt.savefig("output/Best - fitting - 1")
