#!/usr/bin/env python3
import numpy as np
import numpy.fft as fft
import scipy.special as sp
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.optimize import minimize, rosen, rosen_der

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

def cxfunc(par, x):#, c_fraction_3, c_fraction_1
  d, rho = par[0], par[1]
  c = np.zeros_like(x)

  y = np.pi * rho * d**3 / 6.0
  lambda1 = (1 + 2 * y)**2 / (1 - y)**4
  lambda2 = -(1 + 0.5 * y)**2 / (1 - y)**4

  idx = (x <= d)
  x0 = x[idx] / d
  c[idx] = lambda1 + 6 * y * lambda2 + 0.5 * y * lambda1 * x0**3
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


def func_fit(xdata,d,rho,dia):#,c_fraction_3,c_fraction_1
  x = xdata/dia
  N = 100000
  kmax = 1e6
  kmin = 1e-6
  rmin = 1e-6
  rmax = 1e6
  y = np.pi * rho * d ** 3 / 6.0
  fcx = lambda x : cxfunc([d, rho], x)#, c_fraction_3, c_fraction_1
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
file = np.loadtxt("disjoint_void_average.txt",unpack = True)

testx = []
testy = []
for x in file[0][10:35]:
    testx.append(float(x))
for y in file[1][10:35]:
    testy.append(float(y))
testx = np.array(testx)
testy = np.array(testy)
#covariance matrix
C = np.loadtxt("/home/epfl/tan/first-code-document/16R/covariance matrice.txt",unpack = True)
C_1 = np.linalg.inv(C)
C_1 = C_1*(100-50-2)/99.0

def X2_fitting( func_fit, testx, testy, p0):
    accuracy = 20
    p00_min = p0[0] * 0.9
    p00_max = p0[0] * 1.1
    p01_min = p0[1] * 0.9
    p01_max = p0[1] * 1.1
    p02_min = p0[2] * 0.9
    p02_max = p0[2] * 1.1
    X = []
    for i in range(accuracy):
        p0[2] = p02_min + (p02_max - p02_min) / accuracy
        for j in range(accuracy):
            p0[1] = p01_min + (p01_max - p01_min) / accuracy
            for k in range(accuracy):
                p0[0] = p00_min + (p00_max - p00_min)/accuracy
                y = func_fit(file[0], p0[0], p0[1], p0[2])
                f = 0
                for rr in file[0]:
                    if rr / p0[2] < 1:
                        y[f] = -1
                        f = f + 1
                chi_2 = np.dot(np.dot(np.transpose(file[1] - y), C_1),(file[1] - y))
                X.append([chi_2,p0])
                print(i,j,k)
    return Xm

def minimize_X2_fitting(p0,p1,p2):
    y = func_fit(file[0], p0, p1, p2)
    f = 0
    for rr in file[0]:
        if rr / p2 < 1:
            y[f] = -1
            f = f + 1
    chi_2 = np.dot(np.dot(np.transpose(file[1] - y), C_1), (file[1] - y))
    return chi_2
#X = X2_fitting(func_fit,testx, testy, p0 = [0.96818866569206330,46714981600390654,40.583867707955726])
p_0 = [0.968,0.467,40.584]
X = minimize(minimize_X2_fitting, 0.968,0.467,40.584, method='BFGS')
#X_2_min = np.min(X[0])
#ind = X[0].index(X_2_min)
#p0 = X[1][ind]
print(X)


#popt, pcov = curve_fit(func_fit,testx, testy, p0 = [1,0.19900193*6/np.pi,40,1.0,1.0],  maxfev = 2000) #
#print (popt)


plt.figure()
#dia =  38.77299903
plt.errorbar(np.array(file[0]), np.real(np.array(file[1])), yerr=np.array(file[2]), label = "data")
plt.fill_between(np.array(file[0]), np.real(np.array(file[1])) - np.array(file[2]), np.real(np.array(file[1]))+np.array(file[2]), color='red', alpha=0.2)
#r,xi = func_fit(1,0.19900193*6/np.pi)

x = np.linspace(np.min(file[0]),np.max(file[0]),100)
xi = func_fit(x, p0[0], p0[1], p0[2])
#xi = func_fit(x,popt[0],popt[1],popt[2],popt[3],popt[4])#
i = 0
for rr in x:
    if rr/p0[2] < 1:
        xi[i] = -1
        i = i+1
plt.plot(x,xi, label = "model")


#plt.plot(r,xi*1.08573511, label = "model")
#plt.plot(r,xi, label = "model")

#np.savetxt("/home/epfl/tan/first-code-document/fitting_correlation_function_disjoint_void_average_4.txt", np.transpose([x,xi]), fmt = '%g')

#file_power_spectrum = np.loadtxt("xi_disjoint_R16.dat",unpack = True)
#plt.errorbar(np.array(file_power_spectrum[0]), np.array(file_power_spectrum[1]), yerr=np.array(file_power_spectrum[2]), label = "correlation function from disjoint power spectrum")
#plt.fill_between(np.array(file_power_spectrum[0]), np.real(np.array(file_power_spectrum[1])) - np.array(file_power_spectrum[2]), np.real(np.array(file_power_spectrum[1]))+np.array(file_power_spectrum[2]), color='red', alpha=0.2)

plt.legend(loc="upper right")
#plt.ylim(-0.07, 0.07)
#plt.xlim(60, 150)
#plt.savefig("/home/epfl/tan/first-code-document/correlation function chi2/Cx - 3")
plt.savefig("/home/epfl/tan/first-code-document/correlation function chi2/Cx - 6")
plt.show()

#1   [ 1.00415326  0.5453792  43.4646128   1.13503289  1.1787348 ]
#2   0.9681886656920633 0.46714981600390654 40.583867707955726
