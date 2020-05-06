#!/usr/bin/env python3
import numpy as np
import numpy.fft as fft
import scipy.special as sp
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

def windowfn(x, dlnxleft=0.46, dlnxright=0.46):
  xmin = min(x)
  xmax = max(x)
  xleft = np.exp(np.log(xmin) + dlnxleft)
  xright = np.exp(np.log(xmax) - dlnxright)
  w = np.zeros_like(x)
  w[(x > xleft) & (x < xright)] = 1

  il = (x < xleft) & (x > xmin)
  ir = (x > xright) & (x < xmax)

  rl = (x[il] - xmin) / (xleft - xmin)
  rr = (xmax - x[ir]) / (xmax - xright)
  w[il] = rl - np.sin(np.pi * 2 * rl) / (2 * np.pi)
  w[ir] = rr - np.sin(np.pi * 2 * rr) / (2 * np.pi)
  return w

def calc_Mellnu(tt, alpha, q=0):
  n = q - 1 - 1j * tt
  intjlttn = 2**(n-1) * np.sqrt(np.pi) * \
             np.exp(sp.loggamma((1+n)/2.0) - sp.loggamma((2-n)/2.0))
  A = alpha**(1j * tt - q)
  return A * intjlttn

def calc_phi(pk, k0, N, L, q):
  k = k0 * np.exp(np.arange(0,N) * 2 * np.pi / L)
  P = pk(k)
  kpk = (k / k0)**(3-q) * P * windowfn(k)
  phi = np.conj(fft.rfft(kpk)) / L
  phi *= windowfn(k)[len(k) - len(phi):]
  return phi

def xicalc(pk, N=1000, kmin=1e-4, kmax=1e-4, r0=1e-4):
  '''Arguments:
  pk: callable
  N: number of grids for FFT
  kmin, kmax: k range
  r0: minimum r value (~1/kmax)
  '''
  qnu = 1.95
  N2 = int(N / 2) + 1
  k0 = kmin
  G = np.log(kmax / kmin)
  alpha = k0 * r0
  L = 2 * np.pi * N / G

  tt = np.arange(0, N2) * 2 * np.pi / G
  rr = r0 * np.exp(np.arange(0, N) * (G / N))
  prefac = k0**3 / (np.pi * G) * (rr / r0)**(-qnu)

  Mellnu = calc_Mellnu(tt, alpha, qnu)
  phi = calc_phi(pk, k0, N, L, qnu)

  xi = prefac * fft.irfft(phi * Mellnu, N) * N
  return rr, xi


#########################################################
#                     hard sphere                       #
#########################################################

def cxfunc(par, x, c_fraction_3, c_fraction_1):#
  d, rho = par[0], par[1]
  c = np.zeros_like(x)

  y = np.pi * rho * d**3 / 6.0
  lambda1 = (1 + 2 * y)**2 / (1 - y)**4
  lambda2 = -(1 + 0.5 * y)**2 / (1 - y)**4

  idx = (x <= d)
  x0 = x[idx] / d
  c[idx] = lambda1 + 6 * y * lambda2 * x0*c_fraction_1 + 0.5 * y * lambda1 * x0**3 *c_fraction_3#
  return -c

"""
N = 100000
kmax = 1e6
kmin = 1e-6
rmin = 1e-6
rmax = 1e6

d = 1.0
rho = 0.41#0.57295779513082320877
dia = 39
y = np.pi * rho * d**3 / 6.0
"""


def func_fit(xdata,d,rho,dia,c_fraction_3,c_fraction_1):#
  x = xdata/dia
  N = 100000
  kmax = 1e6
  kmin = 1e-6
  rmin = 1e-6
  rmax = 1e6
  y = np.pi * rho * d ** 3 / 6.0
  fcx = lambda x : cxfunc([d, rho], x, c_fraction_3, c_fraction_1)
  k, ck = xicalc(fcx, N, rmin, rmax, kmin)
  nkmin = k[0]
  nkmax = k[-1]
  hk = 8 * np.pi**3 * ck / (1 - 24 * y * ck * 2 * np.pi**2)
  hkint = interp1d(k, hk, kind='cubic')
  fhk = lambda k : hkint(k)
  r, xi = xicalc(fhk, N, kmin, kmax, 0.01)
  gx = interp1d(r, xi, kind='cubic')
  #return r,xi
  return gx(x)

"""
fcx = lambda x : cxfunc([d, rho], x)
k, ck = xicalc(fcx, N, rmin, rmax, kmin)
nkmin = k[0]
nkmax = k[-1]
hk = 8 * np.pi**3 * ck / (1 - 24 * y * ck * 2 * np.pi**2)
hkint = interp1d(k, hk, kind='cubic')
fhk = lambda k : hkint(k)
r, xi = xicalc(fhk, N, kmin, kmax, 0.01)
"""
#hkint = interp1d(np.log(k), hk, kind='cubic')
#fhk = lambda k : hkint(np.log(k))


#np.savetxt('test.txt', np.transpose([k, ck, hk]), fmt='%.8g')

#np.savetxt('test.txt', np.transpose([r, xi]), fmt='%.8g')

#x = np.logspace(-3,3,1000)
#np.savetxt('test.txt', np.transpose([x, fcx(x)]), fmt='%.8g')

#fit the data
#file = np.loadtxt("disjoint_void_average.txt",unpack = True)
file = np.loadtxt("/hpcstorage/zhaoc/share/ting/2PCF_disjoint_voids/correlation_function_auto_disjoint_20/average.dat",unpack = True)
testx = []
testy = []
for x in file[0][10:35]:#:[0:-1]
    testx.append(float(x))
for y in file[1][10:35]:#:[0:-1]
    testy.append(float(y))

file_corre = np.loadtxt("/hpcstorage/zhaoc/share/ting/powspec_fitting/HSfit_16_18/correlation function.dat",unpack = True)

#testx = file[0]
#testy = file[1]               0.19900193*6/np.pi
popt, pcov = curve_fit(func_fit,testx, testy, p0 = [[1,0.45739905,48,1.0,1.0]],  maxfev = 20000) #
print (popt)


plt.figure(figsize=(20,20))
#dia =  38.77299903
plt.errorbar(np.array(file[0]), np.real(np.array(file[1])), yerr=np.array(file[2]), label = "auto correlation function of disjoint voids radius cut $20Mpch^{-1}$")#bins
plt.fill_between(np.array(file[0]), np.real(np.array(file[1])) - np.array(file[2]), np.real(np.array(file[1]))+np.array(file[2]), color='red', alpha=0.2)
#r,xi = func_fit(1,0.19900193*6/np.pi)
x = np.linspace(np.min(file[0]),np.max(file[0]),100)
#xi = func_fit(np.array(file[0]),popt[0],popt[1],popt[2],popt[3])
xi = func_fit(x,popt[0],popt[1],popt[2],popt[3],popt[4])#
#test = func_fit(x,1,0.45,35,1,1)
i = 0
for rr in x:
    if rr/popt[2] < 1:
        xi[i] = -1
        i = i+1
plt.plot(x,xi, label = "Fitting from the Hard Sphere Model")
#plt.plot(file_corre[0],file_corre[1], label = "from powspec 16 - 18 ")
#print (popt[0],popt[1],popt[2])#,popt[3]

#plt.plot(x,test, label = "test")
#plt.plot(r,xi, label = "model")

#np.savetxt("/home/epfl/tan/first-code-document/fitting_correlation_function_disjoint_void_average_4.txt", np.transpose([x,xi]), fmt = '%g')

#file_power_spectrum = np.loadtxt("xi_disjoint_R16.dat",unpack = True)
#plt.errorbar(np.array(file_power_spectrum[0]), np.array(file_power_spectrum[1]), yerr=np.array(file_power_spectrum[2]), label = "correlation function from disjoint power spectrum")
#plt.fill_between(np.array(file_power_spectrum[0]), np.real(np.array(file_power_spectrum[1])) - np.array(file_power_spectrum[2]), np.real(np.array(file_power_spectrum[1]))+np.array(file_power_spectrum[2]), color='red', alpha=0.2)

plt.tick_params(labelsize=23)
plt.legend(loc="lower right", fontsize=20)
plt.xlabel("$r[Mpch^{-1}]$",fontsize=30)
plt.ylabel(r"$\xi$(r)",fontsize=30)
#plt.ylim(-1.1, 0.65)
#plt.xlim(0.1, 200)
plt.ylim(-0.1, 0.2)
plt.xlim(45, 200)
#plt.savefig("/home/epfl/tan/first-code-document/correlation function chi2/Cx - 3")
plt.savefig("/home/epfl/tan/first-code-document/correlation function chi2/Cx - 9 - 4")
plt.show()

#1   [ 1.00415326  0.5453792  43.4646128   1.13503289  1.1787348 ]
#2   0.9681886656920633 0.46714981600390654 40.583867707955726
