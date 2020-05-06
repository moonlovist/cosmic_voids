import math
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from scipy.misc import derivative
from scipy import integrate
import sympy
from scipy.optimize import curve_fit
from sympy import diff


file= pd.DataFrame(np.loadtxt("BDM_nw_R16.dat"))
y = interp1d(np.log(file[0]), np.log(file[1])) #,kind = 'cubic'
xx = np.array(np.log(file[0]))
E = []

file1= pd.DataFrame(np.loadtxt("Albert_Pnw.dat"))
print (len(file[0]))
y1 = interp1d(np.log(file1[0]), np.log(file1[1])) #,kind = 'cubic'
xx1 = np.array(np.log(file1[0]))
EE = []

s = np.linspace(0.0001,5,500)
for S in s:
    f = lambda x: np.exp(y(x))*np.sin(x*S)*x/(S*4*np.pi**2)
    g = lambda x: np.exp(y1(x))*np.sin(x*S)*x/(S*4*np.pi**2)
    E.append(integrate.quad(f,xx[0],xx[479]))
    EE.append(integrate.quad(g,xx1[0],xx1[4999]))
pd.DataFrame(E).to_csv("data_void.csv")
pd.DataFrame(EE).to_csv("data_model.csv")
E = np.array(pd.DataFrame(E)[0])
EE = np.array(pd.DataFrame(EE)[0])
f1 = np.polyfit(s, E, 15)
p1 = np.poly1d(f1)
print (p1)
yvals = p1(s)

f2 = np.polyfit(s, EE, 15)
p2 = np.poly1d(f2)
print (p2)
yvals2 = p2(s)

dif = derivative(p2,s)
dif_func = np.polyfit(s, dif, 15)
dif_func_2 = np.poly1d(dif_func)

fit = np.polyfit(EE,E,5)
fitt = np.poly1d(fit)
print (fitt)
yvals3 = fitt(EE)

def fit_func(x, a, b,c):
    return a*p2(x)+b*dif_func_2(x)+c
popt, pcov = curve_fit(fit_func, s, E)
fit_y = fit_func(s, popt[0], popt[1], popt[2])
plt.figure()
plt.plot(s,fit_y, label ="fitting_function")
plt.plot(s,np.multiply(s,dif/1000).tolist(), label ="derivation_model")
plt.plot(s,yvals2, label ="model_after_fitting")
plt.plot(s,yvals, label ="Void_after_fitting")
plt.plot(s,yvals3, label ="Void_model_fitting")
E1 = np.multiply(np.array(s),E).tolist()
E2 = np.multiply(np.array(s),np.array(E1)).tolist()
plt.plot(s,pd.DataFrame(E)[0], label ="Void_before_fitting")
plt.plot(s,pd.DataFrame(EE)[0], label ="model_before_fitting")
plt.legend(loc="upper right")
plt.savefig("comparision.png")
plt.show()