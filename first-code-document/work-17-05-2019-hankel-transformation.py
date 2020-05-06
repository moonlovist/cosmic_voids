import math
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from scipy.misc import derivative
from scipy import integrate
import sympy
from sympy import diff
from sympy import *
from scipy.fftpack import fft,ifft
from scipy.optimize import curve_fit
import hankel
from hankel import HankelTransform
from scipy.interpolate import InterpolatedUnivariateSpline as spline
#this code aims at proving the methode of Fourier transformation of C(x) to find the total correlation function

#parameter
d = 1
r = np.linspace(0,5,1000,endpoint=False)[1:]
k = np.logspace(-10,20,1000)
#n = 0.008
#mu = n*d**3
"""
mu = 0.7
y = np.pi*mu/6.0
lam1 = (1+2*y)**2 /((1.0-y)**4)
lam2 = - (1+y/2.0)**2 /((1.0-y)**4)
"""
#variable
x = r/d


#equation
#C = lambda x: (-(lam1*y*x**3 /2.0 + 6*y*lam2*x + lam1) for 0<x<1 && 0 for 1<x)
'''def C(x):
    if 1 >= x >=0 :
        return -(lam1*y*x**3 /2.0 + 6*y*lam2*x + lam1)
    elif x > 1:
        return 0
C1 = []
for x1 in x:
    C1.append(C(x1))

k = np.arange(1,1001)
c1 = (-4*np.pi*lam1/(k**3))*(np.sin(k)-k*np.cos(k))
c2 = (-24*np.pi*y*lam2/(k**4))*(2*k*np.sin(k)-(k**2 -2)*np.cos(k)-2)
c3 = (-2*np.pi*y*lam1/(k**6))*((-k**4 +12*k**2-24)*np.cos(k)+(4*k**3 -24*k)*np.sin(k) +24)
Ck_2 = c1+c2+c3
Hk_2 = Ck_2/(1-n*Ck_2)
H_2 = ifft(Hk_2)
x1 = np.linspace(0,20,1000,endpoint=False)

Ck = fft(C1)
print (len(Ck))

Hk = Ck/(1-n*Ck)
H = ifft(Hk)'''
#print ("C(x):",C1)
#print ("C(k):",Ck)
#print ("H(x):",H)
#X = fft(x**3)
#le = np.arange(len(X))


#add andrei's result
file = pd.DataFrame(np.loadtxt("disjoint_void_average.txt"))
d = 42.0

# fit code
def C(x,y,lam1,lam2):
    if 1 >= x >= 0:
        return -(lam1 * y * x ** 3 / 2.0 + 6 * y * lam2 * x + lam1)
    elif x > 1:
        return 0

def H_3(x,n,mu,dia):
    y = np.pi * mu / 6.0
    lam1 = (1 + 2 * y) ** 2 / ((1.0 - y) ** 4)
    lam2 = - (1 + y / 2.0) ** 2 / ((1.0 - y) ** 4)
    x = x/dia
    C3 = lambda x: -(lam1 * y * x ** 3 / 2.0 + 6 * y * lam2 * x + lam1) if (1 >= x.any() >= 0) else 0
    C2 = []
    for x2 in x:
        C2.append(C3(x2))
    h = HankelTransform(nu=0, N=1000, h=0.005)
    Ck_3 = h.transform(C3, k, ret_err=False)
    Hk_3 = Ck_3 / (1 - n * Ck_3)
    #Hk_3 = spline(k, Hk_3)  # Define a spline to approximate transform
    #Hx_3 = h.transform(Hk_3, x, False, inverse=True)
    #Fk = ht.transform(C3, k, ret_err=False)
    #Ck_3 = fft(C2)
    return Hk_3

#np.real(k*ifft(Hk_3))
#print (type(np.real(np.array(file[1]))))
#print(round(np.real(H_3(np.array(file[0])[7:]/32,0.01,0.5,0.2))[1],3))
"""
y = np.pi * mu / 6.0
lam1 = (1 + 2 * y) ** 2 / ((1.0 - y) ** 4)
lam2 = - (1 + y / 2.0) ** 2 / ((1.0 - y) ** 4)
"""

testx = []
testx2 = []
testy = []
#print (file[0])
for x in file[0][7:]:
    testx.append(float(x/d))
for x in file[0][12:]:
    testx2.append(float(x))
for y in file[1][12:]:
    testy.append(float(y))
#popt, pcov = curve_fit(H_3, testx2, testy, p0 = [ 0.008, 0.74, 0.49, 42.0], maxfev = 2000)

#popt, pcov = curve_fit(H_3, np.array(file[0])[7:]/32,np.real(np.array(file[1])[7:]))
#print (popt)
#print (file[0]/42.0)
plt.figure()
plt.plot(k,H_3(r,0.008,0.74,42.0))
#plt.plot(np.array(file[0])[7:]/42,np.real(np.array(file[1])[7:]), label ="nonrecon-disjoint-vv-nz-16R")
#plt.plot(np.array(file[0])/42.0,np.real(np.array(file[1])), label ="nonrecon-disjoint-vv-nz-16R")

#plt.errorbar(np.array(file[0])/42.0, np.real(np.array(file[1])), yerr=np.array(file[2]))
#plt.fill_between(np.array(file[0])/42.0, np.real(np.array(file[1])) - np.array(file[2]), np.real(np.array(file[1]))+np.array(file[2]), color='red', alpha=0.2)

#plt.plot(np.arange(int(len(H_4(file[0],0.008,0.74,0.49,42.0))/2)),H_4(file[0],0.008,0.74,0.49,42.0)[:int(len(H_4(file[0],0.008,0.74,0.49,42.0))/2)],label ="H(x)_2")

#x = np.linspace(0,5,100,endpoint=False)

#plt.plot(x,H_3(x,0.008,0.6,0.2,1),label ="H(x)")
#print (np.real(np.array(file[1])))
#print (np.real(H_3(file[0],popt[0],popt[1],popt[2],popt[3])))
#plt.plot(file[0]/42,np.real(H_3(file[0], 0.008, 0.74, 0.49, 42.0)),label ="H_3(x)")

#plt.plot(file[0]/popt[3],np.real(H_3(file[0],popt[0],popt[1],popt[2],popt[3])),label ="H_4(x)")

#plt.xlim(0.8, 5)
#plt.ylim(-0.5, 1)
#plt.ylim(-0.075, 0.025)
#plt.xscale("log")
#plt.yscale("log")
plt.legend(loc="upper right")
plt.savefig("Hankel Transformation - ps - 1")
plt.show()