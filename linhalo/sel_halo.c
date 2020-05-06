#include "linhalo.h"
#include <ctype.h>//TEST
#include <omp.h>

/* Quicksort the fist N imaginary elements of mesh. */
#ifdef DOUBLE_PREC
void qsort_dens(fftw_complex *mesh, const size_t N)
#else
void qsort_dens(fftwf_complex *mesh, const size_t N)
#endif
{
  const int M = 7;
  const int NSTACK = 64;
  long istack[NSTACK];
  long i, ir, j, k, jstack, l;
  real a, tmp;

  jstack = -1;
  l = 0;
  ir = N - 1;

  for (;;) {
    if (ir - l < M) {
      for (j = l + 1; j <= ir; j++) {
        a = mesh[j][1];
        for (i = j - 1; i >= l; i--) {
          if (mesh[i][1] <= a) break;
          mesh[i + 1][1] = mesh[i][1];
        }
        mesh[i + 1][1] = a;
      }
      if (jstack < 0) break;
      ir = istack[jstack--];
      l = istack[jstack--];
    }
    else {
      k = (l + ir) >> 1;
      SWAP(mesh[k][1], mesh[l + 1][1], tmp);
      if (mesh[l][1] > mesh[ir][1]) {
        SWAP(mesh[l][1], mesh[ir][1], tmp);
      }
      if (mesh[l + 1][1] > mesh[ir][1]) {
        SWAP(mesh[l + 1][1], mesh[ir][1], tmp);
      }
      if (mesh[l][1] > mesh[l + 1][1]) {
        SWAP(mesh[l][1], mesh[l + 1][1], tmp);
      }
      i = l + 1;
      j = ir;
      a = mesh[l + 1][1];
      for (;;) {
        do i++; while (mesh[i][1] < a);
        do j--; while (mesh[j][1] > a);
        if (j < i) break;
        SWAP(mesh[i][1], mesh[j][1], tmp);
      }
      mesh[l + 1][1] = mesh[j][1];
      mesh[j][1] = a;
      jstack += 2;
      if (jstack >= NSTACK) {
        P_EXT("NSTACK for qsort is too small.\n");
        exit(ERR_OTHER);
      }
      if (ir - i + 1 >= j - 1) {
        istack[jstack] = ir;
        istack[jstack - 1] = i;
        ir = j - 1;
      }
      else {
        istack[jstack] = j - 1;
        istack[jstack - 1] = l;
        l = i;
      }
    }
  }
}

#ifdef DOUBLE_PREC
void select_dens(fftw_complex *mesh, const int Ng, const int lowNg,
    const size_t Nh)
#else
void select_dens(fftwf_complex *mesh, const int Ng, const int lowNg,
    const size_t Nh)
#endif
{
  size_t i, j, k, ri, rj, rk, idx, idx2, Ntot, Ntot2, imax;
  int r, ii, jj, kk, ni, nj, nk;
  real tmp;

  printf("\n Re-binning densities ... ");
  fflush(stdout);
  Ntot2 = (size_t) lowNg * lowNg * lowNg;
  /* Generate the low resolution density field using the imaginary part. */
#ifdef OMP
#pragma omp parallel for
#endif
  for (i = 0; i < Ntot2; i++) mesh[i][1] = 0;

  r = Ng / lowNg;
#ifdef OMP
#pragma omp parallel for private(j,i,ii,jj,kk,idx,idx2)
#endif
  for (k = 0; k < Ng; k++) {
    kk = k / r;
    for (j = 0; j < Ng; j++) {
      jj = j / r;
      for (i = 0; i < Ng; i++) {
        ii = i / r;
        idx = MESH_IDX(i, j, k);
        idx2 = ii + jj * lowNg + (size_t) kk * lowNg * lowNg;
#ifdef OMP
#pragma omp atomic
#endif
        mesh[idx2][1] += mesh[idx][0];
      }
    }
  }

#ifdef OMP
#pragma omp parallel for
#endif
  for (i = 0; i < Ntot2; i++) mesh[i][1] /= (r * r * r);

  printf("\r  The density field with a lower resolution is generated.\n"
      "  Selecting largest densities ... ");
  fflush(stdout);
  Ntot = (size_t) Ng * Ng * Ng;

  /* Use the rest of the imaginary part as a heap,
     to store the Nhalo largest densities. */
#ifdef OMP
#pragma omp parallel for
#endif
  for (i = 0; i < Nh; i++) mesh[Ntot2 + i][1] = mesh[i][1];
  qsort_dens(mesh + Ntot2, Nh);

  for (i = Nh; i < Ntot2; i++) {
    if (mesh[i][1] > mesh[Ntot2][1]) {
      mesh[Ntot2][1] = mesh[i][1];
      for (j = 0;;) {
        k = (j << 1) + 1;
        if (k > Nh - 1) break;
        if (k != Nh - 1 && mesh[Ntot2 + k][1] > mesh[Ntot2 + k + 1][1]) ++k;
        if (mesh[Ntot2 + j][1] <= mesh[Ntot2 + k][1]) break;
        SWAP(mesh[Ntot2 + k][1], mesh[Ntot2 + j][1], tmp);
        j = k;
      }
    }
  }

  tmp = mesh[Ntot2][1];
  printf("\r  The %zuth largest density: %g.\n  "
      "Sampling the original field ... ", Nh, tmp);
  fflush(stdout);

#ifdef OMP
#pragma omp parallel for private(jj,ii,ri,rj,rk,idx2,imax,ni,nj,nk,i,j,k,idx)
#endif
  for (kk = 0; kk < lowNg; kk++) {
    rk = kk * r;
    for (jj = 0; jj < lowNg; jj++) {
      rj = jj * r;
      for (ii = 0; ii < lowNg; ii++) {
        ri = ii * r;
        idx2 = ii + jj * lowNg + (size_t) kk * lowNg * lowNg;
        imax = Ntot;

        if (mesh[idx2][1] < tmp) {
          for (nk = 0; nk < r; nk++) {
            k = rk + nk;
            for (nj = 0; nj < r; nj++) {
              j = rj + nj;
              for (ni = 0; ni < r; ni++) {
                i = ri + ni;
                idx = MESH_IDX(i, j, k);
                mesh[idx][0] = 0;
              }
            }
          }
        }
        else {
          for (nk = 0; nk < r; nk++) {
            k = rk + nk;
            for (nj = 0; nj < r; nj++) {
              j = rj + nj;
              for (ni = 0; ni < r; ni++) {
                i = ri + ni;
                idx = MESH_IDX(i, j, k);
                if (imax == Ntot) imax = idx;
                else if (mesh[idx][0] > mesh[imax][0]) {
                  mesh[imax][0] = 0;
                  imax = idx;
                }
                else mesh[idx][0] = 0;
              }
            }
          }
        }
      }
    }
  }
}

#ifdef DOUBLE_PREC
int save_halo(const char *fname, fftw_complex *mesh, const int Ng,
    const double Lbox)
#else
int save_halo(const char *fname, fftwf_complex *mesh, const int Ng,
    const double Lbox)
#endif
{
  FILE *fp;
  int i, j, k, m, n;
  size_t idx;
  double x, y, z;
  real thres;
  char *buf, *end, *cache;

  printf("\n  Filename : %s.\n", fname);
  if (!(fp = fopen(fname, "w"))) {
    P_ERR("cannot write to file `%s'.\n", fname);
    return ERR_FILE;
  }
  thres = 0;

#ifdef OMP
#pragma omp parallel private(buf,end,cache,n) shared(fp)
{
#endif
  buf = calloc(CHUNK, sizeof *buf);
  cache = calloc(MAX_LEN_LINE, sizeof *cache);
  if (!buf || !cache) {
    P_EXT("failed to allocate memory for writing outputs.\n");
    exit(ERR_MEM);
  }
  end = buf;

#ifdef OMP
#pragma omp for private(j,i,x,y,z,idx,m)
#endif
  for (k = 0; k < Ng; k++) {
    z = Lbox * k / Ng;
    for (j = 0; j < Ng; j++) {
      y = Lbox * j / Ng;
      for (i = 0; i < Ng; i++) {
        idx = MESH_IDX(i, j, k);

        if (mesh[idx][0] > thres) {
          x = Lbox * i / Ng;
          n = snprintf(cache, MAX_LEN_LINE,
              OFMT_REAL " " OFMT_REAL " " OFMT_REAL "\n", x, y, z);
          if (n < 0 || n >= MAX_LEN_LINE) {
            P_EXT(FMT_KEY(MAX_LEN_LINE)
                " in `define.h' is not large enough.\n");
            exit(ERR_STRING);
          }

          if (end - buf + n < CHUNK) {        /* there is still space in buf */
            m = cnt_strcpy(end, cache, n + 1);
            if (m >= n + 1) {
              P_EXT("unexpected error for writing line:\n%s\n", cache);
              exit(ERR_STRING);
            }
            end += m;
          }
          else {                              /* write buf to file */
#ifdef OMP
#pragma omp critical
            {
#endif
            if (fwrite(buf, sizeof(char) * (end - buf), 1, fp) != 1) {
              P_EXT("failed to write to output:\n%s\n", cache);
              exit(ERR_FILE);
            }
            fflush(fp);
#ifdef OMP
            }
#endif
            m = cnt_strcpy(buf, cache, n + 1);
            if (m >= n + 1) {
              P_EXT("unexpected error for writing line:\n%s\n", cache);
              exit(ERR_STRING);
            }
            end = buf + m;
          }
        }

      }
    }
  }

  if ((n = end - buf) > 0) {
#ifdef OMP
#pragma omp critical
    {
#endif
    if (fwrite(buf, sizeof(char) * n, 1, fp) != 1) {
      P_EXT("failed to write to output:\n%s\n", cache);
      exit(ERR_FILE);
    }
    fflush(fp);
#ifdef OMP
    }
#endif
  }

  free(buf);
  free(cache);

#ifdef OMP
}
#endif

  fclose(fp);
  return 0;
}

size_t cnt_strcpy(char *dest, const char *src, const size_t num) {
  size_t i = 0;
  while (i < num - 1 && src[i] != '\0') {
    dest[i] = src[i];
    i++;
  }
  dest[i] = '\0';
  while (src[i] != '\0') i++;
  return i;
}
