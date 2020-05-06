#!/usr/bin/env python3
import numpy as np
import numpy.fft as fft
import scipy.special as sp
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import pandas as pd

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

def cxfunc(par, x):
  d, rho = par[0], par[1]
  c = np.zeros_like(x)

  y = np.pi * rho * d**3 / 6.0
  lambda1 = (1 + 2 * y)**2 / (1 - y)**4
  lambda2 = -(1 + 0.5 * y)**2 / (1 - y)**4

  idx = (x <= d)
  x0 = x[idx] / d
  c[idx] = lambda1 + 6 * y * lambda2 * x0 + 0.5 * y * lambda1 * x0**3
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


def func_fit(d,rho):
  N = 100000
  kmax = 1e6
  kmin = 1e-6
  rmin = 1e-6
  rmax = 1e6
  y = np.pi * rho * d ** 3 / 6.0
  fcx = lambda x : cxfunc([d, rho], x)
  k, ck = xicalc(fcx, N, rmin, rmax, kmin)
  nkmin = k[0]
  nkmax = k[-1]
  hk = 8 * np.pi**3 * ck / (1 - 24 * y * ck * 2 * np.pi**2)
  hkint = interp1d(k, hk, kind='cubic')
  fhk = lambda k : hkint(k)
  file1 = np.array(pd.DataFrame(np.loadtxt("Albert_Pnw.dat"))[0])
  Pk = fhk(file1)
  Pk1 = fhk(k)
  r, xi = xicalc(fhk, N, kmin, kmax, 0.01)
  return r, xi, Pk, file1, k, ck, Pk1, hk

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
file2 = pd.DataFrame(np.loadtxt("Albert_Pnw.dat"))
file3 = pd.DataFrame(np.loadtxt("BDM_nw_R16.dat"))
file4 = pd.DataFrame(np.loadtxt("Power-spectrum-disjoint_void_average.txt"))
non_wiggle = interp1d(np.array(file2[0]), np.array(file2[1]), kind='cubic')
y_non_wiggle = non_wiggle(np.array(file4[0]))
print(len(np.array(file4[1])))
print(len(np.array(file3[1])))
#Pk3 = np.array(file3[1])/y_non_wiggle
Pk3 = np.array(file4[1]) - y_non_wiggle
print(np.max(Pk3))
Pk3 = Pk3/(np.max(Pk3)*3)
#hkint = interp1d(np.log(k), hk, kind='cubic')
#fhk = lambda k : hkint(np.log(k))


#np.savetxt('test.txt', np.transpose([k, ck, hk]), fmt='%.8g')

#np.savetxt('test.txt', np.transpose([r, xi]), fmt='%.8g')

#x = np.logspace(-3,3,1000)
#np.savetxt('test.txt', np.transpose([x, fcx(x)]), fmt='%.8g')

#fit the data
file = np.loadtxt("disjoint_void_average.txt",unpack = True)

#file1 = pd.DataFrame(np.loadtxt("Albert_Pnw.dat"))
"""
testx = []
testy = []
for x in file[0][10:]:
    testx.append(float(x/42.0))
for y in file[1][10:]:
    testy.append(float(y))
popt, pcov = curve_fit(func_fit,testx, testy) #, p0 = [0.3,0.3]
print (popt)
"""
plt.figure(figsize=(20,20))
dia = 38.77299903
#plt.errorbar(np.array(file[0])/dia, np.real(np.array(file[1])), yerr=np.array(file[2]), label = "data")
#plt.fill_between(np.array(file[0])/dia, np.real(np.array(file[1])) - np.array(file[2]), np.real(np.array(file[1]))+np.array(file[2]), color='red', alpha=0.2)
r,xi,Pk,xdata,k,ck,Pk1,hk= func_fit(1,0.05*6/np.pi)
#print(k,ck)0.19900193
#r,xi = func_fit(1,0.41)
#plt.plot(r,xi*1.08573511, label = "model")
#np.arange(len(hk))
#plt.plot(k,ck, label = "k-ck")
print(k,hk)
print(xdata)
print(len(k))
print(len(xdata))
nomlize2 = np.sum(Pk)
def rescale_func(x):
  return x
res = []
res1 = []
for x in np.log(np.array(file4[0])):
  res.append(rescale_func(x))

fitt = np.polyfit(np.log(np.array(file4[0])),Pk3,3)

for x in np.log(np.array(file4[0])):
  res1.append(fitt[0]*x**3 + fitt[1]*x**2 + fitt[2]*x + fitt[3])
print ("function:",fitt)
plt.plot(np.log(xdata),Pk, label = "power - spectrum - auto - correlation - mode5")
nomlize = np.sum(Pk3)
#plt.plot(np.log(np.array(file4[0])),Pk3-np.array(res1), label = "power - spectrum voids without cosmology")
plt.plot(np.array(file4[0]),np.array(file4[1])/(np.max(file4[1])*2), label = "power - spectrum -disjoint - voids")

#plt.plot(np.log(np.array(file3[0])),Pk3, label = "Pk3")
# #/()
#plt.plot(np.log(np.array(file3[0])),np.array(res1), label = "power function")
#(100*np.array(res))
#plt.plot(k,hk, label = "power - spectrum - auto - correlation - mode5")
#plt.plot(r,xi, label = "model")
#print (r,xi)
plt.legend(loc="upper left", fontsize=20)
#plt.ylim(-5,5)
#plt.xlim(0.01, 1000)
#plt.xscale("log")
#plt.yscale("log")
plt.savefig("Correction - fitting - power - spectrum - 2")
plt.show()
