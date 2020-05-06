import math
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from scipy.misc import derivative
from scipy import integrate
import sympy

file = pd.DataFrame(np.loadtxt("CATALPTCICz0.562G960S1010008301.dat.2pcf"))
d = 32
plt.figure()
plt.plot(file[0]/d,file[1], label ="nonrecon-disjoint-vv-nz-16R")
#plt.xlim(25, 200)
plt.xlim(0, 5)
plt.ylim(-0.05, 0.25)
plt.legend(loc="upper right")
plt.savefig("fitting-hard_sphere_model-1.png")
plt.show()
np.savetxt("document_vv2fc-3.txt", np.transpose([np.array(file[1])]), fmt = '%g')