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
import scipy
from scipy.fftpack import fft,ifft
from scipy.optimize import curve_fit

#this code aims at proving the methode of Fourier transformation of C(x) to find the total correlation function
#with Bessel function
#parameter
d = 1
r = np.linspace(0,5,100,endpoint=False)
n = 0.008
#mu = n*d**3

mu = 0.8
y = np.pi*mu/6.0
lam1 = (1+2*y)**2 /((1.0-y)**4)
lam2 = - (1+y/2.0)**2 /((1.0-y)**4)

#variable
x = r/d


#equation
#C = lambda x: (-(lam1*y*x**3 /2.0 + 6*y*lam2*x + lam1) for 0<x<1 && 0 for 1<x)
def C(x):
    if 1 >= x >=0 :
        return -(lam1*y*x**3 /2.0 + 6*y*lam2*x + lam1)
    elif x > 1:
        return 0
def Cc(x,S):
    return C(x)*scipy.special.j0(x * S) * x**2/ (4 * np.pi ** 2)
Ck = []
s1 = np.linspace(0.1,100,100)
for S in s1:
    Cc2 = []
    for rs in r:
        #Cc1.append(integrate.quad(Cc,r[0],r[-1]))
        Cc2.append(Cc(rs,S))
    Ck.append(scipy.integrate.trapz(Cc2, r))
#print (Ck)
Ck = np.array(Ck)
Hk = Ck/(1.0-n*Ck)
def Hh(Hk, S, k):
    return Hk*scipy.special.j0(k * S) * k**2/ (4 * np.pi ** 2)
Hx = []
r = np.linspace(0,5,100,endpoint=False)
for x in r:
    Hh2 = []
    for s in s1:
        Hh2.append(Hh(Hk[int(np.argwhere(s1==s))], x, s))
    Hx.append(scipy.integrate.trapz(Hh2, s1))
#print(Hx)
#C1 = []
#for x1 in x:
#    C1.append(C(x1))
'''
k = np.arange(1,1001)
c1 = (-4*np.pi*lam1/(k**3))*(np.sin(k)-k*np.cos(k))
c2 = (-24*np.pi*y*lam2/(k**4))*(2*k*np.sin(k)-(k**2 -2)*np.cos(k)-2)
c3 = (-2*np.pi*y*lam1/(k**6))*((-k**4 +12*k**2-24)*np.cos(k)+(4*k**3 -24*k)*np.sin(k) +24)
Ck_2 = c1+c2+c3
Hk_2 = Ck_2/(1-n*Ck_2)
H_2 = ifft(Hk_2)
x1 = np.linspace(0,20,1000,endpoint=False)
'''
#Ck = fft(C1)
#print (len(Ck))

#Hk = Ck/(1-n*Ck)
#H = ifft(Hk)
#print ("C(x):",C1)
#print ("C(k):",Ck)
#print ("H(x):",H)
#X = fft(x**3)
#le = np.arange(len(X))


#add andrei's result
file = np.loadtxt("CATALPTCICz0.562G960S1010008301.dat.2pcf",unpack = True)
dia = 42

# fit code
"""
def C(x,y,lam1,lam2):
    if 1 >= x >= 0:
        return -(lam1 * y * x ** 3 / 2.0 + 6 * y * lam2 * x + lam1)
    elif x > 1:
        return 0

def H_3(x,n,mu,k):
    y = np.pi * mu / 6.0
    lam1 = (1 + 2 * y) ** 2 / ((1.0 - y) ** 4)
    lam2 = - (1 + y / 2.0) ** 2 / ((1.0 - y) ** 4)

    C2 = []
    for x2 in x:
        C2.append(C(x2,y,lam1,lam2))
    Ck_3 = fft(C2)
    Hk_3 = Ck_3 / (1 - n * Ck_3)
    return np.real(k*ifft(Hk_3))
print (type(np.real(np.array(file[1]))))
print(round(np.real(H_3(np.array(file[0])[7:]/32,0.01,0.5,0.2))[1],3))"""
#popt, pcov = curve_fit(H_3, xData, yData)   , p0 = [0.01,0.5,0.2], maxfev = 2000
#popt, pcov = curve_fit(H_3, np.array(file[0])[7:]/32,np.real(np.array(file[1])[7:]))
#print (popt)[7:]
plt.figure()
#plt.plot(np.array(file[0])/dia,np.real(np.array(file[1])), label ="nonrecon-disjoint-vv-nz-16R")
#l = 0.3
#plt.plot(r,Hx,label ="H(x)")
#plt.plot(np.arange(len(Ck)),Ck) '''sinc function'''
#print (Hk)
#plt.plot(np.arange(len(Hk)),Hk)
#plt.plot(file[0]/32,100*np.real(H_3(file[0]/32,0.01,0.5,0.2)),label ="H_3(x)")
#plt.xlim(0.8, 5)
#plt.ylim(-1, 1)
#plt.plot(s1,Ck)
#plt.xscale("log")
#plt.yscale("log")
#plt.legend(loc="upper right")
plt.savefig("Fourier Transformation method new-4")
plt.show()
