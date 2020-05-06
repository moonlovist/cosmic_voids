#include "linhalo.h"
#include <math.h>
#include <gsl/gsl_rng.h>
#include <gsl/gsl_randist.h>
#include <gsl/gsl_spline.h>
#ifdef OMP
#include <omp.h>
#endif

int init_pk(double *k, double *P, const size_t Nk, const CONF *conf) {
  size_t i;

  if (k[0] > 2 * M_PI / conf->Lbox) {
    P_ERR("the minimum k of the input power spectrum is too large.\n");
    return ERR_RANGE;
  }
  if (k[Nk - 1] < M_PI * conf->Ngrid / conf->Lbox) {
    P_ERR("the maximum k of the input power spectrum is too small.\n");
    return ERR_RANGE;
  }

  for (i = 0; i < Nk; i++) {
    k[i] = log(k[i]);
    P[i] = log(P[i]);
  }

  printf("  Power spectrum interpolated.\n");
  return 0;
}

#ifdef DOUBLE_PREC
int init_field(const int Ngrid, fftw_complex **mesh, fftw_plan *plan)
#else
int init_field(const int Ngrid, fftwf_complex **mesh, fftwf_plan *plan)
#endif
{
  size_t size = (size_t) Ngrid * Ngrid * Ngrid;

  printf("\n  Initialising ... ");
  fflush(stdout);

#ifdef DOUBLE_PREC
  *mesh = fftw_malloc(sizeof(fftw_complex) * size);
#else
  *mesh = fftwf_malloc(sizeof(fftwf_complex) * size);
#endif
  if (!(*mesh)) {
    P_ERR("failed to allocate memory for the density field.\n");
    return ERR_MEM;
  }

#ifdef DOUBLE_PREC
#ifdef OMP
  if (fftw_init_threads() == 0) {
    P_ERR("failed to initialize FFTW with OpenMP.\n");
    return ERR_OTHER;
  }
  fftw_plan_with_nthreads(omp_get_max_threads());
#endif
  *plan = fftw_plan_dft_3d(Ngrid, Ngrid, Ngrid, *mesh, *mesh,
      FFTW_BACKWARD, FFTW_ESTIMATE);
#else
#ifdef OMP
  if (fftwf_init_threads() == 0) {
    P_ERR("failed to initialize FFTW with OpenMP.\n");
    return ERR_OTHER;
  }
  fftwf_plan_with_nthreads(omp_get_max_threads());
#endif

  *plan = fftwf_plan_dft_3d(Ngrid, Ngrid, Ngrid, *mesh, *mesh,
      FFTW_BACKWARD, FFTW_ESTIMATE);
#endif

  printf("\r  The density field is initialised.\n");
  return 0;
}

int gauss_ran_field(const CONF *conf, const double *lnk, const double *lnP,
#ifdef DOUBLE_PREC
    const size_t Nk, fftw_plan *fp, fftw_complex *mesh
#else
    const size_t Nk, fftwf_plan *fp, fftwf_complex *mesh
#endif
    ) {
  int i, j, k;
  size_t idx;
  double fac, norm, fac2, ki, kj, kk, ksq, kv, P;
  int Ng;
  gsl_rng *r;
  gsl_interp_accel *acc;
  gsl_spline *spline;

  Ng = conf->Ngrid;
  fac = 2 * M_PI / conf->Lbox;
  norm = pow(conf->Lbox, -1.5);
//  r = gsl_rng_alloc(gsl_rng_mt19937);
//  gsl_rng_set(r, conf->seed);
  acc = gsl_interp_accel_alloc();
  spline = gsl_spline_alloc(gsl_interp_cspline, Nk);
  gsl_spline_init(spline, lnk, lnP, Nk);

  printf("  Generating Fourier modes ... ");
  fflush(stdout);

  mesh[0][0] = mesh[0][1] = 0;
  /* Generate random randoms with a single thread to be reproducible. */
/*
  size_t Ntot = (size_t) Ng * Ng * Ng;
  for (idx = 1; idx < Ntot; idx++) {
    mesh[idx][0] = gsl_ran_gaussian(r, 1);
    mesh[idx][1] = gsl_ran_gaussian(r, 1);
  }
*/

#ifdef OMP
#pragma omp parallel private(r)
{
#endif
  r = gsl_rng_alloc(gsl_rng_mt19937);
  gsl_rng_set(r, conf->seed + omp_get_thread_num());
#ifdef OMP
#pragma omp for private(j,i,ki,kj,kk,idx,ksq,kv,P,fac2)
#endif
  /* Loop in k,j,i order to reduce cache miss. */
  for (k = 0; k < Ng; k++) {
    kk = (k <= Ng / 2) ? k : k - Ng;
    kk *= fac;
    for (j = 0; j < Ng; j++) {
      kj = (j <= Ng / 2) ? j : j - Ng;
      kj *= fac;
      for (i = 0; i < Ng; i++) {
        ki = (i <= Ng / 2) ? i : i - Ng;
        ki *= fac;

        idx = MESH_IDX(i, j, k);
        if (i == 0 && j == 0 && k == 0) {
          mesh[idx][0] = mesh[idx][1] = 0;
          continue;
        }

        ksq = ki * ki + kj * kj + kk * kk;
        kv = log(ksq) * 0.5;    /* log(sqrt(ksq)) */
        P = gsl_spline_eval(spline, kv, acc);
        P = exp(P * 0.5);       /* sqrt(P) */
        fac2 = P * norm;

        /* NOTE: not reproducible! */
        mesh[idx][0] = gsl_ran_gaussian(r, 1);
        mesh[idx][1] = gsl_ran_gaussian(r, 1);

        mesh[idx][0] *= fac2;
        mesh[idx][1] *= fac2;
      }
    }
  }

  gsl_rng_free(r);
#ifdef OMP
}
#endif

  printf("\r  Density field in Fourier space is generated.\n"
      "  Executing FFT ... ");
  fflush(stdout);

#ifdef DOUBLE_PREC
  fftw_execute(*fp);
  fftw_destroy_plan(*fp);
#ifdef OMP
  fftw_cleanup_threads();
#endif
#else
  fftwf_execute(*fp);
  fftwf_destroy_plan(*fp);
#ifdef OMP
  fftwf_cleanup_threads();
#endif
#endif

  printf("\r  FFT finished successfully.\n");

  gsl_spline_free(spline);
  gsl_interp_accel_free(acc);
  return 0;
}

#ifdef DOUBLE_PREC
int save_dens(const char *fname, fftw_complex *mesh, const int Ng)
#else
int save_dens(const char *fname, fftwf_complex *mesh, const int Ng)
#endif
{
  FILE *fp;
  size_t i, Ntot;

  printf("\n  Filename : %s.\n  Preparing for data ... ", fname);
  fflush(stdout);

  Ntot = (size_t) Ng * Ng * Ng;
#ifdef OMP
#pragma omp parallel for
#endif
  for (i = 0; i < Ntot; i++) mesh[i >> 1][i % 2] = mesh[i][1];

  if (!(fp = fopen(fname, "w"))) {
    P_ERR("cannot write to file `%s'.\n", fname);
    return ERR_FILE;
  }

  if (fwrite(mesh, sizeof(real) * Ntot, 1, fp) != 1) {
    P_EXT("failed to write density field to file `%s'.\n", fname);
    return ERR_FILE;
  }

  printf("\r  The density field is saved.\n");
  fclose(fp);
  return 0;
}
