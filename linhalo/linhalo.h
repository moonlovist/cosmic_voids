/**********************************************************
**                                                       **
**      Generate a linear halo catalogue                 **
**      Author: Cheng Zhao <zhaocheng03@gmail.com>       **
**                                                       **
**********************************************************/

#ifndef _LINHALO_H_
#define _LINHALO_H_

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <fftw3.h>
#include "define.h"
#include "load_conf.h"

int read_pk(const char *, double **, double **, size_t *);

int init_pk(double *, double *, const size_t, const CONF *);

#ifdef DOUBLE_PREC

int init_field(const int, fftw_complex **, fftw_plan *);

int gauss_ran_field(const CONF *, const double *, const double *,
    const size_t, fftw_plan *, fftw_complex *);

void qsort_dens(fftw_complex *, const size_t);

void select_dens(fftw_complex *, const int, const int, const size_t);

int save_halo(const char *, fftw_complex *, const int, const double);

int save_dens(const char *, fftw_complex *, const int);

#else

int init_field(const int, fftwf_complex **, fftwf_plan *);

int gauss_ran_field(const CONF *, const double *, const double *,
    const size_t, fftwf_plan *, fftwf_complex *);

void qsort_dens(fftwf_complex *, const size_t);

void select_dens(fftwf_complex *, const int, const int, const size_t);

int save_halo(const char *, fftwf_complex *, const int, const double);

int save_dens(const char *, fftwf_complex *, const int);

#endif

size_t cnt_strcpy(char *, const char *, const size_t);

#endif
