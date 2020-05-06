import numpy as np
import os
from scipy.interpolate import interp1d


def func_frac(powspec):
  length = len(np.array(powspec[0]))
  for i in range(length):
    if (powspec[0][i]<0.1):
      powspec[1][i] = powspec[1][i]*0.5
    else:
      powspec[1][i] = 0 
  return powspec[1] 
      
file_1 = np.loadtxt("/hpcstorage/zhaoc/share/ting/powspec_fitting/HSfit_16_18/input/Pnw.dat", unpack = True)
file_2 = func_frac(file_1)
files = os.listdir("/hpcstorage/zhaoc/share/ting/powspec_fitting/HSfit_16_18/input/mocks_ori")
files.sort()
for name in files:
  file_5 = np.loadtxt("/hpcstorage/zhaoc/share/ting/powspec_fitting/HSfit_16_18/input/mocks_ori/%s"%name,unpack = True)
  file_4 = file_5[1]# - 1*file_2
  np.savetxt("/hpcstorage/zhaoc/share/ting/powspec_fitting/HSfit_16_18/input/mocks/%s"%name, np.transpose([file_1[0],file_4]), fmt = '%g')




