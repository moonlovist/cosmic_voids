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
#parameters
rho = 0.9 #number density of the particles
R = 1.0
#x = np.linspace(-1,1,100000)
r = np.linspace(1.001, 2.0, 5000)
#mu = rho*R**3
mu = 1.1
#variables
y = np.pi*mu/6.0

s1 = (1-y)**2
s2 = 6*y*(1-y)
s3 = 18*y**2
s4 = -12*y*(1 + 2*y)

#resoudre the euqtion
j = complex(0,1)
w = (-1 + 3**(1/2.0) *j)/2.0
p = (3*s1*s3 - s2**2)/(3.0*s1**2)
q = (27*s1**2 *s4 - 9*s1*s2*s3 + 2*s2**3)/(27.0*s1**3)
print(s1,s2,s3,s4)
print(p)
print(q)
print(abs(-q/2.0) - abs(((q/2.0)**2 + (p/3.0)**3)**(1/2.0)))

print('theta',((q/2.0)**2 + (p/3.0)**3)**(1/2.0))

#mid_term1 = (abs(-q/2.0) + abs(((q/2.0)**2 + (p/3.0)**3)**(1/2.0)))**(1/3.0)
mid_term1 = ((-q/2.0) + abs(((q/2.0)**2 + (p/3.0)**3)**(1/2.0)))**(1/3.0)
print ('mid1', mid_term1)
mid_term2 = ((-q/2.0 - ((q/2.0)**2 + (p/3.0)**3)**(1/2.0)))**(1/3.0)
print(mid_term2)
mid_term3 = - s2/(3*s1)
print ('mid3', mid_term3)
t1 = mid_term1 + mid_term2 + mid_term3
t2 = w*mid_term1 + w**2 *mid_term2 + mid_term3
t3 = w**2 *mid_term1 + w*mid_term2 + mid_term3

#equetions

#def S(t):
    #return s1*t**3 + s2*t**2 + s3*t + s4
def S(t):
    return s1*(t-t1)*(t-t2)*(t-t3)

def S_1or(t):
    return 3*s1*t**2 + 2*s2*t + s3
def S_2or(t):
    return 6*s1*t + 2*s2
def L(t):
    return 12*y*((1 + 0.5*y)*t + 1 + 2*y)
def L_1or(t):
    return 12*y*(1+0.5*y)
def L_2or(t):
    return 0
#def get_root(a, b):
#    bin = (b - a)/1000
    #    x = a
    #    for i in range(1000):
    #        if S(x)*S(x + i*bin)<0:
    #            return x
#        x = x + i*bin
#t0 = get_root(-1,1)

def tfunction1(r, t, ti):
    return (t - ti)*t*L(t)*np.exp(t*(r-R))/S(t)
#def tfunction1(r, ti):
#    return Limit((t - ti)*t*L(t)*np.exp(t*(r-R))/S(t), t, ti)

#def sum1(r):
#    return tfunction1(r, t1) + tfunction1(r, t2) + tfunction1(r, t3)

#def sum1(r):
#    return tfunction1(r, t1, t1) + tfunction1(r, t2, t2) + tfunction1(r, t3, t3)
def sum1(r):
    return t1*L(t1)*np.exp(t1*(r-R))/(s1*(t1-t2)*(t1-t3)) + t2*L(t2)*np.exp(t2*(r-R))/(s1*(t2-t1)*(t2-t3)) + t3*L(t3)*np.exp(t3*(r-R))/(s1*(t3-t1)*(t3-t2))
def g1(r):
    return 1/(12.0*y*r) *sum1(r)

"""calculate first order sum"""
def f1(t,ti):
    return (t - ti)**2
def f1_1or(t,ti):
    return (t - ti)*2
def f2(t):
    return (L(t)/S(t))**2
def f2_1or(t):
    return 1#(2*L(t)*L_1or(t)*S(t)**2 - L(t)**2 *2*S(t)*S_1or(t))/(S(t)**4)
def f3(r,t):
    return np.exp(t*(r - 2*R))
def f3_1or(r,t):
    return (r - 2*R)*np.exp(t*(r - 2*R))

def tfunction2(r, t, ti):
    return f1_1or(t,ti)*t*f2(t)*f3(r,t) + f1(t,ti)*f2(t)*f3(r,t) + f1(t,ti)*t*f2_1or(t)*f3(r,t) + f1(t,ti)*t*f2(t)*f3_1or(r,t)
#function2 = lambda r, t, ti: (t - ti)**2 *t*(L(t)/S(t))**2 *np.exp(t*(r-2*R))
#tfunction2(r, t, ti) = diff(function2, t, 1)

def sum2(r):
    return tfunction2(r, t1+0.00001, t1) + tfunction2(r, t2+0.00001+0.00001*j, t2) + tfunction2(r, t3+0.00001+0.00001*j, t3)
def g2(r):
    return sum2(r)/(12.0*y*r)

#def tfunction3(r, t, ti):
#    return diff((t - ti)**3 *t*(L(t)/S(t))**3 *np.exp(t*(r-3*R)), t, 2)
#function3 = lambda r, t, ti: (t - ti)**3 *t*(L(t)/S(t))**3 *np.exp(t*(r-3*R))
#tfunction3(r, t, ti) = diff(function3, t, 2)

def f_3_1(t,ti):
    return (t - ti)**3
def f_3_1_1or(t,ti):
    return 3*(t - ti)**2
def f_3_1_2or(t,ti):
    return 6*(t - ti)

def f_3_2(t):
    return (L(t)/S(t))**3
def f_3_2_1or(t):
    return (3*L(t)**2 *L_1or(t)*S(t)**3 - 3*L(t)**3 *S(t)**2 *S_1or(t))/(S(t)**6)
def f_3_2_2or(t):
    return (f_3_2_2or_1(t) + f_3_2_2or_2(t) + f_3_2_2or_3(t))/ (S(t)**12)
def f_3_2_2or_1(t):
    return (S(t)**6)*(6*L(t)*L_1or(t)**2 *S(t)**3 + 3*L(t)**2 *L_2or(t) + 3*L(t)**2 *L_1or(t)*3*S(t)**2 *S_1or(t))
def f_3_2_2or_2(t):
    return -(S(t)**6)*(9*L(t)**2 *L_1or(t)*S_1or(t)*S(t)**2 + 3*L(t)**3 *S_2or(t)*S(t)**2 + 3*L(t)**3 *S_1or(t)**2 *2*S(t))
def f_3_2_2or_3(t):
    return -6*S(t)**5 *S_1or(t)*(3*L(t)**2 *L_1or(t)*S(t)**3 - 3*L(t)**3 *S(t)**2 *S_1or(t))

def f_3_3(r,t):
    return np.exp(t*(r - 3*R))
def f_3_3_1or(r,t):
    return (r - 3*R)*np.exp(t*(r - 3*R))
def f_3_3_2or(r,t):
    return (r - 3*R)**2 *np.exp(t*(r - 3*R))

def u1(r, t, ti):
    return f_3_1_2or(t,ti)*t*f_3_2(t)*f_3_3(r,t) + f_3_1_1or(t,ti)*f_3_2(t)*f_3_3(r,t) + f_3_1_1or(t,ti)*t*f_3_2_1or(t)*f_3_3(r,t) + f_3_1_1or(t,ti)*t*f_3_2(t)*f_3_3_1or(r,t)
def u2(r, t, ti):
    return f_3_1_1or(t,ti)*f_3_2(t)*f_3_3(r,t) + f_3_1(t,ti)*f_3_2_1or(t)*f_3_3(r,t) + f_3_1(t,ti)*t*f_3_2(t)*f_3_3_1or(r,t)
def u3(r, t, ti):
    return f_3_1_1or(t,ti)*t*f_3_2_1or(t)*f_3_3(r,t) + f_3_1(t,ti)*f_3_2_1or(t)*f_3_3(r,t) + f_3_1(t,ti)*t*f_3_2_2or(t)*f_3_3(r,t) + f_3_1(t,ti)*t*f_3_2_1or(t)*f_3_3_1or(r,t)
def u4(r, t, ti):
    return f_3_1_1or(t,ti)*t*f_3_2(t)*f_3_3_1or(r,t) + f_3_1(t,ti)*f_3_2(t)*f_3_3_1or(r,t) + f_3_1(t,ti)*t*f_3_2_1or(t)*f_3_3_1or(r,t) + f_3_1(t,ti)*t*f_3_2(t)*f_3_3_2or(r,t)

def tfunction3(r, t, ti):
    return u1(r, t, ti) + u2(r, t, ti) + u3(r, t, ti) + u4(r, t, ti)
def sum3(r):
    return tfunction3(r, t1+0.0001, t1) + tfunction3(r, t2+0.0001, t2) + tfunction3(r, t3+0.0001, t3)
def g3(r):
    return 1/(24.0*y*r) *sum3(r)

def g(r):
    for x in r:
        if R > x >= 0:
            return 0
        if 2*R > x >= R:
            return g1(x)
        if 3*R > x >= 2*R:
            return g1(x)+g2(x)
        if 4*R > x >= 3*R:
            return g1(x)+g2(x)+g3(x)
        #if 4*R > x >= 3*R:
        #    return g3(x)
        else:
            return 0

k = []
for i in range(5000):
    if R > r[i] >= 0:
        k.append(0)
    if 2 *R > r[i] >= 1 *R:
        k.append(abs(g1(r[i])))
    if 3 *R >= r[i] >= 2 *R:
        k.append(abs(g2(r[i]))+abs(g1(r[i])))#
    if 4 *R >= r[i] >= 3 *R:
        k.append(abs(g2(r[i]))+abs(g3(r[i])))#+abs(g1(r[i]))
    #if 4 * R > x >= 3 * R:
    #    g.append(g3(x))
print (t1,t2,t3,mid_term1)
print (f2_1or(t1), f2_1or(t2))
print (f2_1or(t3))
plt.figure()
#plt.plot(r,k, label ="correlation function")
def check(h,r):
    return h(r,t1+0.01,t1)+h(r,t2+0.01+0.01*j,t2)+h(r,t2+0.01+0.01*j,t2)
def g1(r,t,ti):
    return f1_1or(t,ti)*t*f2(t)*f3(r,t)
def g2(r,t,ti):
    return f1(t,ti)*f2(t)*f3(r,t)
def g3(r,t,ti):
    return f1(t,ti)*t*f2_1or(t)*f3(r,t)
def g4(r,t,ti):
    return f1(t,ti)*t*f2(t)*f3_1or(r,t)
#plt.plot(r,check(g1,r)/(12.0*y*r), label ="1")
#plt.plot(r,check(g2,r)/(12.0*y*r), label ="2")
#plt.plot(r,check(g3,r)/(12.0*y*r), label ="3")
#plt.plot(r,check(g3,r)/(12.0*y*r) + check(g1,r)/(12.0*y*r), label ="3")
#plt.plot(r,check(g4,r)/(12.0*y*r), label ="4")
#k.pop(1)
plt.plot(r[:-1],k[:-1], label ="correlation function")
#print (S_1or(r))
#plt.plot(r,S(r), label ="S")
#plt.plot(r,S_1or(r), label ="S'")
plt.legend(loc="upper right")
plt.savefig("hard_sphere_model-13.png")
plt.show()
