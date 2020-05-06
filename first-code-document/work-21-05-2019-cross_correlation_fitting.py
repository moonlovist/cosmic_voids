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
#this code aims at proving the methode of Fourier transformation of C(x) to find the total correlation function
#the analycal function of correlation function
#parameter
d = 1
r = np.linspace(0.001,5,500,endpoint=False)
n = 0.008
j = complex(0,1)
#mu = n*d**3

mu = 0.8
y = np.pi*mu/6.0
lam1 = (1+2*y)**2 /((1.0-y)**4)
lam2 = - (1+y/2.0)**2 /((1.0-y)**4)

#variable
x = r/d


#equation
#C = lambda x: (-(lam1*y*x**3 /2.0 + 6*y*lam2*x + lam1) for 0<x<1 && 0 for 1<x)
def phi1(r,L1,s1,s2,s3):
    tem1 = (s1 - s2)
    tem2 = (s1 - s3)
    tem3 = (s2 - s3)
    tem4 = 1 + L1 *s1
    tem5 = 1 + L1 *s2
    tem6 = 1 + L1 *s3
    return (-np.exp(r*s2) *s2*tem5*(tem2) +
   np.exp(r*s1) *s1*tem4*(tem3) +
   np.exp(r*s3) *(tem1)*s3*tem6)/((tem1)*(tem2)*(tem3))
def phi2(r,L1,s1,s2,s3):
    tem1 = (s1 - s2)
    tem2 = (s1 - s3)
    tem3 = (s2 - s3)
    tem4 = 1 + L1 *s1
    tem5 = 1 + L1 *s2
    tem6 = 1 + L1 *s3
    tem7 = (s2 + s3)
    return ((np.exp(r*s1) *r*s1*tem4**2)/((tem1)**2 *(tem2)**2) + (
 np.exp(r*s2) *r*s2*tem5**2)/((tem1)**2 *(tem3)**2) + (
 np.exp(r*s3) *r*s3*tem6**2)/((tem2)**2 *(tem3)**2) - (
np.exp(r*s1) *tem4*(L1*s1**3 - s2*s3 - s1*(s2 + s3 + 3*L1*s2*s3) +
      s1**2 *(3 + L1*tem7)))/((tem1)**3 *(tem2)**3) - (np.exp(
    r*s3) *tem6*(-s1*(s2 + s3 + 3*L1*s2*s3 - L1*s3**2) +
      s3*(s2*(-1 + L1*s3) + s3*(3 + L1*s3))))/((-tem2)**3 *(-tem3)**3) - (np.exp(r*s2) *tem5*(s2*(L1*s2**2 - s3 + s2*(3 + L1*s3)) +
      s1*(L1*s2**2 - s3 - s2*(1 + 3*L1*s3))))/((-tem1)**3 *(tem3)**3))
def phi3(r,L1,s1,s2,s3):
    tem1 = (s1 - s2)
    tem2 = (s1 - s3)
    tem3 = (s2 - s3)
    tem4 = 1 + L1*s1
    tem5 = 1 + L1*s2
    tem6 = 1 + L1*s3
    tem7 = (s2 + s3)
    return 1/2.0 *((np.exp(r*s1) *r**2 *s1*tem4**3)/((tem1)**3 *(tem2)**3) +
                   (np.exp(r*s2) *r**2 *s2*tem5**3)/((-tem1)**3 *(tem3)**3) +
                   (np.exp(r*s3) *r**2 *s3*tem6**3)/((-tem2)**3 *(-tem3)**3) -
           (2*np.exp(r*s1) *r*tem4**2 *(2*L1*s1**3 - s2*s3 -2*s1*(s2 + s3 + 2*L1*s2*s3) + s1**2 *(5 + L1*tem7)))/((tem1)**4 *(tem2)**4) -
           (2*np.exp(r*s2) *r*tem5**2 *(s2*(2*L1*s2**2 - 2*s3 + s2*(5 + L1*s3)) +s1*(L1*s2**2 - s3 - 2*s2*(1 + 2*L1*s3))))/((tem1)**4 *(tem3)**4) -
           (2*np.exp(r*s3) *r*tem6**2 *(-s1*(s2 + 4*L1*s2*s3 + s3*(2 - L1*s3)) +s3*(s2*(-2 + L1*s3) + s3*(5 + 2*L1*s3))))/((tem2)**4 *(tem3)**4) +
           (6*np.exp(r*s2)*tem5*(s1**2 *tem6*(s2 + s3 + 2*L1*s2*s3) +s2*(L1**2 *s2**4 - 4*s2*s3 + s3**2 + s2**2 *(5 - L1*s3) +L1*s2**3 *(5 + L1*s3)) +
            s1*(L1**2 *s2**4 + s3**2 + s2*s3*(-1 + 3*L1*s3) -L1*s2**3 *(1 + 5*L1*s3) - 2*s2**2 *(2 + 5*L1*s3))))/((-tem1)**5 *(tem3)**5) +
           (6*np.exp(r*s1)*tem4*(L1**2 *s1**5 + s2*s3*(s2 + s3 + L1*s2*s3) +
        L1*s1**4 *(5 + L1*tem7) - s1**3 *(-5 + 5*L1**2 *s2*s3 + L1*tem7) -2*s1**2 *(2*s3 + s2*(2 + 5*L1*s3)) +
        s1*(s3**2 + s2*s3*(-1 + 3*L1*s3) +s2**2 *(1 + 3*L1*s3 + 2*L1**2 *s3**2))))/((tem1)**5 *tem2**5) +
           (6*np.exp(r*s3)*tem6*(s1**2 *tem5*(s2 + s3 + 2*L1*s2*s3) +s3*(s2**2 + s2*s3*(-4 - L1*s3 + L1**2 *s3**2) + s3**2 *(5 + 5*L1*s3 + L1**2 *s3**2)) +
        s1*(s2**2 *(1 + 3*L1*s3) + s3**2 *(-4 - L1*s3 + L1**2 *s3**2) - s2*s3*(1 + 10*L1*s3 + 5*L1**2 *s3**2))))/((-tem2)**5 *(-tem3)**5))
def phi4(r,L1,s1,s2,s3):
    tem1 = (s1 - s2)
    tem2 = (s1 - s3)
    tem3 = (s2 - s3)
    tem4 = 1 + L1 *s1
    tem5 = 1 + L1 *s2
    tem6 = 1 + L1 *s3
    tem7 = (s2 + s3)
    return 1/6 *((np.exp(r *s1) *r**3 *s1* tem4**4)/(tem1**4 *tem2**4) +
                 (np.exp(r *s2) *r**3 *s2* tem5**4)/(tem1**4 *tem3**4) +
                 (np.exp(r *s3) *r**3 *s3* tem6**4)/(tem2**4 *tem3**4) -

                 1 / ((-tem2)**5 *(-tem3)** 5) *3 *np.exp(r *s3) *r**2 *
                 tem6**3*(-s1 *(s2 + 5 *L1 *s2 *s3 + s3 *(3 - L1 *s3)) +
                           s3 *(s2 *(-3 + L1 *s3) + s3 *(7 + 3 *L1 *s3))) -
                 (3 *np.exp(r *s2) *r**2 *tem5 **3 *(s2 *(3 *L1 *s2**2 - 3 *s3 + s2 *(7 + L1 *s3)) +
                 s1 *(L1 *s2**2 - s3 - s2 *(3 + 5 *L1 *s3)))) / ((-tem1)**5 *tem3**5)+

                 1 / (tem1**6 *tem3**6) *12 *np.exp(r *s2) *r *
                 tem5**2 *(s2 *(3 *L1**2 *s2**4 - 12 *s2 *s3 + 3 *s3**2 +
                                    s2**2 *(14 - 4 *L1 *s3) + 2 *L1 *s2**3 *(7 + L1 *s3)) +
                              2 *s1 *(L1**2 *s2**4 + s3**2 + 4 *L1 *s2 *s3**2 -
                                        6 *s2**2 *(1 + 2 *L1 *s3) -2 *L1 *s2**3 *(1 + 3 *L1 *s3)) +
                              s1**2 *(tem6) *(2 *s3 + s2 *(3 + 5 *L1 *s3))) -
                 (3 *np.exp(r *s1) *r**2 *tem4 **3 *(3 *L1 *s1**3 - s2 *s3 + s1**2 *(7 + L1 *tem7) -
                 s1 *(3 *s3 + s2 *(3 + 5 *L1 *s3)))) / (tem1**5 *tem2**5) +

                 1 / (tem2**6 *tem3**6) *12 *np.exp(r *s3) *r *
                 tem6**2 *(s1**2 *tem5 *(3 *s3 + s2 *(2 + 5 *L1 *s3)) +
                              2 *s1 *(-6 *L1 *s2 *s3**2 *(2 + L1 *s3) +
                                        s2**2 *(1 + 4 *L1 *s3) +
                                        s3**2 *(-6 - 2 *L1 *s3 + L1**2 *s3**2)) +
                              s3 *(3 *s2**2 + 2 *s2 *s3*(-6 - 2 *L1 *s3 + L1**2 *s3**2) +
                                    s3**2 *(14 + 14 *L1 *s3 + 3 *L1**2 *s3**2)))+

                 1 / (tem1**6 *tem2**6) *12 *np.exp(r *s1) *r *
                 tem4**2 *(3 *L1**2 *s1**5 +
                              2 *s2 *s3 *(s2 + s3 + L1 *s2 *s3) -
                              12 *s1**2 *(s2 + s3 + 2 *L1 *s2 *s3) +
                              2 *L1 *s1**4 *(7 + L1 *tem7) -
                              2 *s1**3 *(-7 + 6 *L1**2 *s2 *s3 + 2 *L1 *tem7) +
                              s1 *(3 *s3**2 + 8 *L1 *s2 *s3**2 +
                                    s2**2 *(3 + 8 *L1 *s3 + 5 *L1**2 *s3**2))) +

                 1 / ((-tem1)**7*(tem3)**7) *12 *np.exp(r *s2) *(tem5) *(-s2 *(5 *L1**3 *s2**6 - 5 *s3**3 +
                                                    s2**3 *(42 - 58 *L1 *s3) - 5 *L1 *s2**4 *(-14 + L1 *s3) -
                                                    3 *s2 *s3**2*(-9 + L1 *s3) +
                                                    5 *L1**2 *s2**5 *(7 + L1 *s3) +
                                                    3 *s2**2 *s3 *(-18 + 7 *L1 *s3)) -
                                             s1**2 *(tem6) *(-8 *s3**2 + s2 *s3 *(11 - 19 *L1 *s3) +
                                                                 7 *L1 *s2**3 *(3 + 5 *L1 *s3) +
                                                                 s2**2 *(27 + 58 *L1 *s3 - 5 *L1**2 *s3**2)) +
                                             s1**3 *(tem6) *(s3 *(5 + 3 *L1 *s3) +
                                                                 L1 *s2**2 *(3 + 5 *L1 *s3) +
                                                                 s2 *(5 + 14 *L1 *s3 + 5 *L1**2 *s3**2)) +
                                             s1 *(-5 *L1**3 *s2**6 + 5 *s3**3 +
                                                   5 *L1**2 *s2**5 *(1 + 7 *L1 *s3) +
                                                   s2 *s3**2 *(-11 + 19 *L1 *s3) +
                                                   L1 *s2**4 *(58 + 133 *L1 *s3) +
                                                   s2**3 *(54 + 98 *L1 *s3 - 56 *L1**2 *s3**2) +
                                                   s2**2 *s3 *(-18 - 85 *L1 *s3 + 8 *L1**2 *s3**2)))-

                 1 /(tem1**7 *tem2**7) *12 *np.exp(r *s1)*tem4 *(5 *L1**3 *s1**7 +
                                          5 *L1**2 *s1**6 *(7 + L1 *tem7) -
                                          5 *L1 *s1**5 *(-14 + 7 *L1**2 *s2 *s3 + L1 *tem7) -
                                          s1**4 *(-42 + 133 *L1**2 *s2 *s3 + 58 *L1 *tem7) -
                                          s2 *s3 *(5 *s3**2 + 8 *s2 *s3 *tem6 +
                                                  s2**2 *(5 + 8 *L1 *s3 + 3 *L1**2 *s3**2)) +
                                          s1**3 *(3 *s3 *(-18 + 7 *L1 *s3) +
                                                     7 *L1 *s2**2 *(3 + 8 *L1 *s3 + 5 *L1**2 *s3**2) +
                                                     s2 *(-54 - 98 *L1 *s3 + 56 *L1**2 *s3**2)) +
                                          s1**2 *(-3 *s3**2 *(-9 + L1 *s3) +
                                                     s2 *s3 *(18 + 85 *L1 *s3 - 8 *L1**2 *s3**2) -
                                                     L1 *s2**3 *(3 + 8 *L1 *s3 + 5 *L1**2 *s3**2) +
                                                     s2**2 *(27 + 85 *L1 *s3 + 53 *L1**2 *s3**2 -
                                                                5 *L1**3 *s3**3)) -
                                          s1 *(5 *s3**3 + s2 *s3**2 *(-11 + 19 *L1 *s3) +
                                                s2**2 *s3*(-11 + 8 *L1 *s3 + 19 *L1**2 *s3**2) +
                                                s2**3 *(5 + 19 *L1 *s3 + 19 *L1**2 *s3**2 + 5 *L1**3 *s3**3))) +

                 1 / ((-tem2)**7 *(-tem3)**7) *12 *np.exp(r *s3) *(tem6) *(s1 **3 *(tem5) *(s3 *(5 + 3 *L1 *s3) + L1 *s2**2 *(3 + 5 *L1 *s3) +
                                                           s2 *(5 + 14 *L1 *s3 + 5 *L1**2 *s3**2)) +
                                             s1**2 *(tem5) *(-3 *s3**2 *(9 + 7 *L1 *s3) +
                                                                 s2**2 *(8 + 19 *L1 *s3 + 5 *L1**2 *s3**2) -
                                                                 s2 *s3 *(11 + 58 *L1 *s3 + 35 *L1**2 *s3**2)) -
                                             s3 *(-s2**3 *(5 + 3 *L1 *s3) + 3 *s2**2 *s3 *(9 + 7 *L1 *s3) +
                                                   s2 *s3 **2 *(-54 - 58 *L1 *s3 - 5 *L1**2 *s3**2 +
                                                        5 *L1**3 *s3**3) +
                                                   s3**3 *(42 + 70 *L1 *s3 + 35 *L1**2 *s3**2 +
                                                              5 *L1**3 *s3**3)) +
                                             s1 *(s2**3 *(5 + 19 *L1 *s3 + 8 *L1**2 *s3**2) -
                                                   s2**2 *s3 *(11 + 85 *L1 *s3 + 56 *L1**2 *s3**2) +
                                                   s3**3 *(54 + 58 *L1 *s3 + 5 *L1**2 *s3**2 -
                                                              5 *L1**3 *s3**3) +
                                                   s2 *s3**2 *(-18 + 98 *L1 *s3 + 133 *L1**2 *s3**2 +
                                                        35 *L1**3 *s3**3))))

#get the roots
def LL1(e):
    return (1+e/2.0)/(1.0+2*e)
def SS1(e):
    return -3*e/(2.0*(1 + 2*e))
def SS2(e):
    return -(1-e)/(2.0*(1 + 2*e))
def SS3(e):
    return -(1-e)**2/(12.0*e*(1 + 2*e))
def midcalculation1(S1, S2, S3):
    return (-4*(S2**2 - 3*S1*S3)**3 + (2*S2**3 - 9*S1*S2*S3 +27*S3**2)**2)**(1/2.0)
def midcalculation2(S1, S2, S3):
    return (-2*S2**3 + 9*S1*S2*S3 - 27*S3**2 + midcalculation1(S1,S2,S3))**(1/3.0)
def ss1(S1, S2, S3):
    return -(S2/(3*S3)) - (2**(1/3.0)*(-S2**2 + 3*S1*S3))/(3*S3*midcalculation2(S1, S2, S3)) + 1/(3*2**(1/3.0)*S3) *midcalculation2(S1, S2, S3)
def ss2(S1, S2, S3):
    return -(S2/(3*S3)) + ((1+j*np.sqrt(3))*(-S2**2 + 3*S1*S3))/(3*2**(2/3.0) *S3*midcalculation2(S1, S2, S3)) - \
           (1-j*np.sqrt(3))/(6*2**(1/3.0)*S3) *midcalculation2(S1, S2, S3)
def ss3(S1, S2, S3):
    return -(S2/(3*S3)) + ((1-j*np.sqrt(3))*(-S2**2 + 3*S1*S3))/(3*2**(2/3.0) *S3*midcalculation2(S1, S2, S3)) - \
           (1+j*np.sqrt(3))/(6*2**(1/3.0)*S3) *midcalculation2(S1, S2, S3)
def theta(r):
    for x in r:
        if x>0:
            return 1
        else:
            return 0
#the correlation function
def g(x, e, c, d):
    k = []
    for r in x/d:
        k.append(-1/(12.0*e*r) *(1/(SS3(e)) *phi1(r-1,LL1(e),ss1(SS1(e),SS2(e),SS3(e)),ss2(SS1(e),SS2(e),SS3(e)),ss3(SS1(e),SS2(e),SS3(e)))*np.heaviside(r-1,0)+
                        1/(SS3(e)**2) *phi2(r-2,LL1(e),ss1(SS1(e),SS2(e),SS3(e)),ss2(SS1(e),SS2(e),SS3(e)),ss3(SS1(e),SS2(e),SS3(e)))*np.heaviside(r-2,0)+
                        1/(SS3(e)**3) *phi3(r-3,LL1(e),ss1(SS1(e),SS2(e),SS3(e)),ss2(SS1(e),SS2(e),SS3(e)),ss3(SS1(e),SS2(e),SS3(e)))*np.heaviside(r-3,0)+
                        1/(SS3(e)**4) *phi4(r-4,LL1(e),ss1(SS1(e),SS2(e),SS3(e)),ss2(SS1(e),SS2(e),SS3(e)),ss3(SS1(e),SS2(e),SS3(e)))*np.heaviside(r-4,0)))
    return np.real((np.array(k)-1)*c)
"""
print(abs(g(0.5,r)))
print (SS1(0.5),SS2(0.5),SS3(0.5),LL1(0.5),midcalculation1(SS1(0.5),SS2(0.5),SS3(0.5)),midcalculation2(SS1(0.5),SS2(0.5),SS3(0.5)))
print (ss1(SS1(0.5),SS2(0.5),SS3(0.5)),ss2(SS1(0.5),SS2(0.5),SS3(0.5)),ss3(SS1(0.5),SS2(0.5),SS3(0.5)))
print(phi1(1.5,LL1(0.5),ss1(SS1(0.5),SS2(0.5),SS3(0.5)),ss2(SS1(0.5),SS2(0.5),SS3(0.5)),ss3(SS1(0.5),SS2(0.5),SS3(0.5))))
print(phi2(1.5,LL1(0.5),ss1(SS1(0.5),SS2(0.5),SS3(0.5)),ss2(SS1(0.5),SS2(0.5),SS3(0.5)),ss3(SS1(0.5),SS2(0.5),SS3(0.5))))
print(phi3(1.5,LL1(0.5),ss1(SS1(0.5),SS2(0.5),SS3(0.5)),ss2(SS1(0.5),SS2(0.5),SS3(0.5)),ss3(SS1(0.5),SS2(0.5),SS3(0.5))))
print(phi4(1.5,LL1(0.5),ss1(SS1(0.5),SS2(0.5),SS3(0.5)),ss2(SS1(0.5),SS2(0.5),SS3(0.5)),ss3(SS1(0.5),SS2(0.5),SS3(0.5))))
"""
#print(phi4(r,LL1(0.5),ss1(SS1(0.5),SS2(0.5),SS3(0.5)),ss2(SS1(0.5),SS2(0.5),SS3(0.5)),ss3(SS1(0.5),SS2(0.5),SS3(0.5))))
#the direct correlation function
"""
def C(x):
    if 1 >= x >=0 :
        return -(lam1*y*x**3 /2.0 + 6*y*lam2*x + lam1)
    elif x > 1:
        return 0
C1 = []
for x1 in x:
    C1.append(C(x1))
"""
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
file = np.loadtxt("disjoint_void_average.txt",unpack = True)
dia = 42.0

#add cross correlation function
file_cross = np.loadtxt("cross-disjoint_void_average.txt",unpack = True)

# fit code
"""
def C(x,y,lam1,lam2):
    if 1 >= x >= 0:
        return -(lam1 *y *x**3 / 2.0 + 6 *y *lam2 *x + lam1)
    elif x > 1:
        return 0

def H_3(x,n,mu,k):
    y = np.pi *mu / 6.0
    lam1 = (1 + 2 *y)**2 / ((1.0 - y)**4)
    lam2 = - (1 + y / 2.0)**2 / ((1.0 - y)**4)

    C2 = []
    for x2 in x:
        C2.append(C(x2,y,lam1,lam2))
    Ck_3 = fft(C2)
    Hk_3 = Ck_3 / (1 - n *Ck_3)
    return np.real(k*ifft(Hk_3))
print (type(np.real(np.array(file[1]))))float(np.real(np.array(file[1])[7:])
print(round(np.real(H_3(np.array(file[0])[7:]/32,0.01,0.5,0.2))[1],3))"""
#popt, pcov = curve_fit(H_3, xData, yData)   , p0 = [0.01,0.5,0.2], maxfev = 2000
"""
print (file[1])
print (type(file[1][0]))
print (type(file[0][0]))

#file[0][7:]/32.0file[1][7:]"""
dia1 = 32
testx = []
testy = []
for x in file_cross[0][6:]:
    testx.append(float(x))
for y in file_cross[1][6:]:
    testy.append(float(y))
popt, pcov = curve_fit(g,testx, testy, p0 = [0.16,0.19,32.0])
print (popt)
#print (g(testx,0.3,0.3))
plt.figure()
print(file[0])
#plt.plot(testx,g(testx,0.3,0.3), label ="test function")
#plt.plot(np.array(file[0])/dia,np.real(np.array(file[1])), label ="nonrecon-disjoint-vv-nz-16R")

"""
#plot auto correlation
plt.errorbar(np.array(file[0])/dia, np.real(np.array(file[1])), yerr=np.array(file[2]), label = "voids auto-correlation function")
plt.fill_between(np.array(file[0])/dia, np.real(np.array(file[1])) - np.array(file[2]), np.real(np.array(file[1]))+np.array(file[2]), color='red', alpha=0.2)
"""


#plot cross correlation
plt.errorbar(np.array(file_cross[0]), np.real(np.array(file_cross[1])), yerr=np.array(file_cross[2]), label = "voids-halos cross-correlation function")
plt.fill_between(np.array(file_cross[0]), np.real(np.array(file_cross[1])) - np.array(file_cross[2]), np.real(np.array(file_cross[1]))+np.array(file_cross[2]), color='red', alpha=0.2)

#print (file[1])
#print (file[2])
#l = 0.3
#plt.plot(x,H*l,label ="H(x)")
#plt.plot(np.arange(len(Ck)),Ck) '''sinc function'''
#print (1/(SS3(0.5)) *phi1(r-1,LL1(0.5),ss1(SS1(0.5),SS2(0.5),SS3(0.5)),ss2(SS1(0.5),SS2(0.5),SS3(0.5)),ss3(SS1(0.5),SS2(0.5),SS3(0.5)))*theta(r-1))
#r = [r[2]]

#print (r)
#print(abs(g(0.5,r)))
#plt.plot(r,abs(phi2(r,LL1(0.5),ss1(SS1(0.5),SS2(0.5),SS3(0.5)),ss2(SS1(0.5),SS2(0.5),SS3(0.5)),ss3(SS1(0.5),SS2(0.5),SS3(0.5)))*np.heaviside(r-2,0)))
x = np.linspace(0.001,np.max(file_cross[0]))
h = np.real(g(x, popt[0], popt[1], popt[2]))
i = 0
for rr in x:
    if rr/popt[2] < 1:
        h[i] = -1
        i = i+1



plt.plot(x,h, label ="model")
#plt.plot(r,np.real(g(r,0.16,0.19)), label ="model-1")
#plt.plot(file[0]/32,100*np.real(H_3(file[0]/32,0.01,0.5,0.2)),label ="H_3(x)")
#plt.xlim(0.001, 5)
#plt.xlim(0.8, 3.5)
plt.xlim(40, 100)
#plt.ylim(-0.075, 0.025)
plt.ylim(-0.015, 0.03)
#plt.ylim(-0.05, 0.05)
#plt.ylim(-1.2, 1.2)
plt.legend(loc="upper right")
plt.savefig("cross correlation function fitting-7")
plt.show()