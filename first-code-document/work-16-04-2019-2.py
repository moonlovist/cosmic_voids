import math
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from scipy.misc import derivative
from scipy import integrate
import scipy
import sympy
plt.figure()
file = pd.DataFrame(np.loadtxt("k13t90.hhh"))
f = interp1d(file[0], file[1]) #,kind = 'cubic'
s = np.linspace(1.001,5,500)
ss = np.linspace(0.1,0.9,100)
list(s).extend(list(ss))
s = np.array(s)
y = f(s)


file1= pd.DataFrame(np.loadtxt("BDM_nw_R16.dat"))
y1 = interp1d(np.log(file1[0]), np.log(file1[1]), kind = 'cubic') #
x1 = np.array(np.log(file1[0]))
E = []


file2= pd.DataFrame(np.loadtxt("avg_16R18.2pcf"))
plt.plot(np.array(file2[0]), file2[1]*(np.array(file2[0]))**2, label ="void_16R18")


s1 = np.linspace(0.1,150,100)
for S in s1:
    g = lambda x: np.exp(y1(x))*scipy.special.j0(np.exp(x)*S)*(np.exp(x))**3/(4*np.pi**2) #*np.exp(-np.exp(x)**2)
    """E.append(integrate.quad(g,x1[0],x1[-1]))"""
    E.append(scipy.integrate.trapz(g(x1), x1))
E = np.array(pd.DataFrame(E)[0])
np.savetxt("document_voids_1.txt", np.transpose([s1,E]), fmt = '%g')
#plt.plot(s1, E, label ="void_before_fitting")

#E2 = []
#p = 0.9
#n = lambda x: np.pi*p*x**3 /6
#h = lambda n: (1-n/2)/(1-n)**3
#h = lambda x: 1 - (3*x/4) + (x**3 /16)
#plt.plot(s1,h(s1), label ="function")

plt.plot(s,y, label ="model_before_fitting")
plt.legend(loc="upper right")
plt.savefig("hard_sphere_model-2.png")
plt.show()