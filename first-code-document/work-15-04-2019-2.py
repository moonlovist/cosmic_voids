import math
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from scipy.misc import derivative
from scipy import integrate
import sympy

file= pd.DataFrame(np.loadtxt("Albert_Pnw.dat"))
y = interp1d(np.log(file[0]), np.log(file[1])) #,kind = 'cubic'
y1 = np.exp(y(np.log(file[0])))
#s = sympy.symbols('s')
xx = np.array(np.log(file[0]))
#print (x)
k = []
k1 = []
#for a in x:
    #k.append(np.sin(a))
    #k1.append(math.pow(np.pi,2))
#print (type(k[0]))
#print (type(k1[0]))
#k2 = (np.array(k)).astype(type('float', (float,), {}))
E = []
s = np.linspace(1,4,500)
for S in s:
    f= lambda x: np.exp(y(x))*np.sin(x*S)*x/(S*4*np.pi**2)
    #a = xx[0].astype(type('float', (float,), {}))
    #b = xx[4999].astype(type('float', (float,), {}))
    E.append(integrate.quad(f,xx[0],xx[4999]))
plt.figure()
print (E)
E1 = np.multiply(np.array(s),np.array(pd.DataFrame(E)[0])).tolist()
E2 = np.multiply(np.array(s),np.array(E1)).tolist()
plt.plot(s,pd.DataFrame(E)[0])
plt.savefig("ex6.png")
plt.show()