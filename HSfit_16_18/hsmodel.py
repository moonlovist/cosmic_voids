#!/usr/bin/env python3
import numpy as np
from FFTlog import xicalc

def cxfunc(y, x):
  d0 = 1.0
  c = np.zeros_like(x)

  lambda1 = (1 + 2 * y)**2 / (1 - y)**4
  lambda2 = -(1 + 0.5 * y)**2 / (1 - y)**4

  idx = (x <= d0)
  x0 = x[idx] / d0
  c[idx] = (lambda1 + 6 * y * lambda2 * x0 + 0.5 * y * lambda1 * x0**3)
  return -c

def hkfunc(par, kmin, kmax, Nk):
  d, rho, A = par[0], par[1], par[2]
  d0 = 1.0
  y = np.pi * rho * d0**3 / 6.0

  xmin = 1.0 / kmax
  xmax = 1.0 / kmin

  fcx = lambda x : cxfunc(y, x)
  k, ck = xicalc(fcx, int(Nk), xmin, xmax, kmin)

  hk = 8 * np.pi**3 * ck / (1 - 24 * y * ck * 2 * np.pi**2)
  k /= d
  hk *= A

  return [k, hk]

