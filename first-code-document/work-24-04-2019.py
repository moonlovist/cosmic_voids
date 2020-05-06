import math
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from scipy.misc import derivative
from scipy import integrate
import sympy

p = 0.9
g = lambda x:1 + y1(x)*p + y2(x)*p**2

def y1(x):
    if x<2:
        return np.pi/12 *(x+4)* (x-2)**2
    else:
        return 0



def c2(x):
    if 1 > x >= 0:
        return np.pi**2/630 *(x**6 - 63*x**4 + 315*x**2 - 525)
    elif 3 > x >= 1:
        return -np.pi**2/(1260*x) *(x**3 + 12*x**2 + 27*x - 6)*(x - 3)**4
    elif x >= 3:
        return 0



def t1(x):
    if 1 > x >= 0:
        return -np.pi**2/1260 *(x**6 - 63*x**4 - 70*x**3 + 315*x**2 + 525*x - 1050)
    elif 2 > x >= 1:
        return np.pi**2/(1260*x) *(x**5 + 4*x**4 - 51*x**3 - 10*x**2 + 479*x - 81)
    elif x >= 2:
        return 0

#t2(x) = t1(x)

def E1(x):
    if x >= 0:
        return X0(x)
    elif np.sqrt(3) > x >= 1:
        return -1/2 *y1(x)**2 + np.pi/2 *X1(x)
    elif 2 >= x >= np.sqrt(3):
        return -1/2 *y1(x)**2
    elif x > 2:
        return 0

#v1(x) = y1(x)

def X0(x):
    return np.pi/2 *(np.pi*(-3*x**6 /560.0 + x**4 /15.0 - x**3 /9.0 - x**2 /2.0 + 22*x/15.0 - 5/6.0 - 9/(35.0*x))
                     + (-3*x**4 /280.0 + 41*x**2 /420.0 )*np.sqrt(3 - x**2)
                     + (-23*x /15.0 + 36.0/(35.0*x))*np.arccos(x/np.sqrt(12.0 - 3*x**2))
                     + (3*x**6 /560.0 - x**4 /15.0 + x**2*0.5*np.arccos((x**2 -2)/(4.0-x**2)))
                     + (2*x/15.0 - 9/(35.0*x))*np.arccos((-12 + 11*x**2 -2*x**4)/(12.0 - 3*x**2)))

def X1(x):
    return (-3*x**4 /280.0 + 41*x**2 /420.0)*np.sqrt(3 - x**2) \
           + (-23*x /15.0 + 36/(35.0*x))*np.arccos(x/np.sqrt(12 - 3*x**2)) \
           +(3*x**6 /560.0 - x**4 /15 + x**2 /2.0 + 2*x/15.0 - 9/(35.0*x))*np.arccos((x**2 + x - 3)/np.sqrt(12 - 3*x**2)) \
           +(3*x**6 /560.0 - x**4 /15 + x**2 /2.0 - 2*x/15.0 + 9/(35.0*x))*np.arccos((-x**2 + x - 3)/np.sqrt(12 - 3*x**2))
y2 = lambda x: c2(x) + 2*t1(x) + E1(x) + 1/2.0 *y1(x)**2
x = np.linspace(0.1,2,500)
plt.figure()
h = []
for s in x:
    h.append(y2(s))
plt.plot(x,h, label ="model")
plt.legend(loc="upper right")
plt.savefig("hard_sphere_model-7.png")
plt.show()