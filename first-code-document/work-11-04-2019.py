import math
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
plt.figure()
file= pd.DataFrame(np.loadtxt("BDM_nw_R16.dat"))
filevoid = pd.DataFrame(np.loadtxt("Albert_Pnw.dat"))
filehalo = pd.DataFrame(np.loadtxt("BDM_nw.dat"))
plt.plot(file[0], file[1], label = "voids power spectrum")
plt.plot(filevoid[0], filevoid[1], label = "linear power spectrum non wiggle")
plt.plot(filehalo[0], filehalo[1], label = "linear power spectrum")
plt.xscale("log")
plt.yscale("log")
plt.legend(loc="upper right")
plt.savefig("ex3e.png")
plt.show()


