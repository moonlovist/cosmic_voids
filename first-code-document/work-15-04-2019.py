import math
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from scipy.misc import derivative

plt.figure()
file= pd.DataFrame(np.loadtxt("Albert_Pnw.dat"))
y = interp1d(np.log(file[0]), np.log(file[1])) #,kind = 'cubic'
y1 = y(np.log(file[0]))
y2 = np.exp(y(np.log(file[0])))
print(y(np.log(file[0])[315:320]))
dif = derivative(y,np.log(file[0])[318:4682])
plt.plot(file[0][318:4682],np.exp(dif))
plt.plot(file[0],y2)
plt.xscale("log")
plt.yscale("log")
plt.savefig("ex2.png")
plt.show()