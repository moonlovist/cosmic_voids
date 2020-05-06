import math
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from scipy.misc import derivative
from scipy import integrate
import sympy

#parameters
d = 16.0 #(separation distance)
#n = 1 #(number density of particles)
#yita = 1
#a = 1
#b = 1

#variables
r = np.linspace(3,30,1000) #(chosen region)
#x = r/d #(dimensionless distance)
#y = np.pi*n*d**3 /6.0 #(reduced density)
rho = 0.005
#d = (rho/n)**(1/3.0)
x = r/d
y = d**3 *rho*np.pi/6.0
lam1 = (1+2*y)**2 /((1.0-y)**4)
lam2 = - (1+y/2.0)**2 /((1.0-y)**4)

u0 = (72/3**(1/2.0))**(1/3.0) *y**(2/3.0)
u1 = (1-y)**2
a0 = 3**(1/2.0) *u1 *(1 + 2*y)**2
b0 = y*(2+y)**3
u2 = (a0 + (a0**2 + b0**2)**(1/2.0))**(1/3.0)
u = u0/(u1*u2)
v2 = (a0 - (a0**2 + b0**2)**(1/2.0))**(1/3.0)
v = u0/(u1*v2)
yita = (u + v)**(1/2.0)
yita1 = yita
yita2 = -yita

j = complex(0,1)
yib1 = -1/2.0 + (j* 3**(1/2.0) /2)
yib2 = -1/2.0 - (j* 3**(1/2.0) /2)
z2 = yib1*u + yib2*v
a = abs(np.real(z2**(1/2.0)))
b = abs(np.imag(z2**(1/2.0)))
a1 = a + b*j
a2 = a + b*j

alpha3 = 24*lam2*y**2
alpha5 = lam2*y**2*6/5.0
phi0 = 2#lam1 + 6*y*lam2 + 1/2.0 *y*lam1
phi1 = 0#lam1 + 12*y*lam2 + 2*y*lam1
phi2 = 0#12*y*lam2 + 6*y*lam1
phi3 = 0#12*y*lam1
phi4 = 0#12*y*lam1 + 6*alpha3*phi0
phi5 = 0#6*alpha3*phi1


#equations
def P(p):
    return phi5 + phi4*p + phi3* p**2 + phi2* p**3 + phi1* p**4 + phi0* p**5 - 6*alpha3*phi1 - 6*alpha3*phi0

def Q(p):
    return p**6 - 6*alpha3*p**2 - 120*alpha5

def phi(x):
    return x*lam1 + 6*y*lam2*x**2 + y*lam1*x**4/2.0

def C(x):
    return -(lam1 + 6*y*lam2*x + y*lam1*x**3/2.0)

def g1(x):
    return A1*np.exp(yita*(x-1))

def g2(x):
    return A2*np.exp(-yita*(x-1))

def g3(x):
    return 2*np.exp(a*(x-1))*(np.real(B1)*np.cos(b*(x-1)) - np.imag(B1)*np.sin(b*(x-1)))

def g4(x):
    return 2*np.exp(-a*(x-1))*(np.real(B2)*np.cos(b*(x-1)) - np.imag(B2)*np.sin(b*(x-1)))

def g(x):
    return  (g1(x) + g2(x) + g3(x)+ g4(x))/x#

A1 = P(yita1)/derivative(Q, yita1)
A2 = P(yita2)/derivative(Q, yita2)
B1 = P(a1)/derivative(Q, a1)
B2 = P(a2)/derivative(Q, a2)

plt.figure()
plt.plot(x,g(x), label ="model")
#plt.plot(x,phi(x), label ="direct correlation")
plt.legend(loc="upper right")
plt.savefig("hard_sphere_model-10.png")
plt.show()