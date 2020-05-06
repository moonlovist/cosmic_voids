#!/usr/bin/env python3
import os
import sys
import numpy as np

def get_icov(input_mocks, cov_file, compute_cov=True, save_cov=False):
  '''Read/Compute the inverse of the covariance matrix.
  Return: [Nmock, icov], where icov is the inverse of the covariance matrix.'''
  if compute_cov:
    # Read the list of P(k) from mocks
    mocks = []
    with open(input_mocks) as f:
      for line in f:
        fname = line.rstrip('\n')
        if fname != '':
          mocks.append(fname)
    Nmock = len(mocks)

    # Read P(k) of mocks
    pkmock = [None] * Nmock
    for i in range(Nmock):
      pkmock[i] = np.loadtxt("/hpcstorage/zhaoc/share/ting/powspec_fitting/HSfit_4/input/mocks/%s"%mocks[i], usecols=(1,), unpack=True)
    pkmock = np.array(pkmock)

    # Compute the covariance matrix
    mean = np.mean(pkmock, axis=0)
    pkmock -= mean
    cov = np.dot(pkmock.T, pkmock)
    icov = np.linalg.inv(cov)

    if save_cov:
      np.savetxt(cov_file, icov, header=str(Nmock))
  else:         # compute_cov = False
    with open(cov_file, 'r') as f:
      Nmock = int(f.readline())
    icov = np.loadtxt(cov_file, skiprows=1)

  return [Nmock, icov]

def get_range(k, Pk, fitmin, fitmax, nmock, icov):
  '''Return [k, Pk, icov] within the fitting range.'''
  if len(k) != icov.shape[0]:
    raise Exception('bin size of data and mocks do not match.')
  idx = (k >= fitmin) & (k <= fitmax)
  k_fit = k[idx]
  nbin = len(k_fit)
  if nbin < 1:
    raise Exception('cannot find enough bins for the fitting.')
  if nmock < nbin + 3:
    raise Exception('number of mocks is not enought for the fitting.')

  P_fit = Pk[idx]
  icov_fit = icov[np.ix_(idx,idx)] * (nmock - nbin - 2.0) / (nmock - 1.0)
  return [k_fit, P_fit, icov_fit]

