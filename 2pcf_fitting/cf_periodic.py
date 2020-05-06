#!/usr/bin/env python
import numpy as np
import sys

n = len(sys.argv)
if n != 3:
  print 'Usage: %s DD_file BOXSIZE order' % sys.argv[0]
  sys.exit(1)

try:
  boxsize = float(sys.argv[2])
  if boxsize <= 0:
    print 'Error: the boxsize is not set correctly!'
    sys.exit(1)
except:
  print 'Error: cannot recognize the boxsize!'
  sys.exit(1)

try:
  d = np.loadtxt(sys.argv[1])
except:
  print 'Error: cannot read the input DD file!'
  sys.exit(1)

if d.shape[1] != 4 and d.shape[1] != 6:
  print 'Error: cannot recognize the format of the DD file!'
  sys.exit(1)

x1 = d[:,0]
x2 = d[:,1]
x = (x1 + x2) * 0.5
n = len(x)

rr = 4 * np.pi / 3.0 * (x2**3 - x1**3) / boxsize**3
mono = d[:,3] / rr - 1 
if d.shape[1] == 4:
#  np.savetxt(output, np.transpose([x,mono]), fmt='%.8g')
  for i in range(n):
    print x[i], mono[i]
else:
  nmu = 100
  mu = np.linspace(0.5, nmu - 0.5, nmu) / nmu
  rrquad = rr * (2.5 * (3 * mu**2 - 1)).sum() / nmu
  quad = (d[:,5] - rrquad) / rr
  np.savetxt("/hpcstorage/zhaoc/share/ting/2PCF_disjoint_voids/correlation_function_auto_disjoint_20/100.dat", np.transpose([x,mono,quad]), fmt='%.8g')
  #print 
  for i in range(n):
    print x[i], mono[i], quad[i]

