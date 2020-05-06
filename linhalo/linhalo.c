#include "load_conf.h"
#include "linhalo.h"

int main(int argc, char *argv[]) {
  int ecode;
  double *k, *P;
  size_t Nk;
  CONF conf;
#ifdef DOUBLE_PREC
  fftw_complex *mesh;
  fftw_plan fp;
#else
  fftwf_complex *mesh;
  fftwf_plan fp;
#endif

  printf("Loading configurations ... ");
  fflush(stdout);
  if ((ecode = read_conf(argc, argv, &conf))) {
    P_EXT("failed to load the configurations.\n");
    return ecode;
  }
  if ((ecode = check_conf(&conf))) {
    P_EXT("please check your configurations.\n");
    return ecode;
  }
  print_conf(&conf);
  printf(FMT_DONE);

  printf("Processing the linear power spectrum ... ");
  fflush(stdout);
  if ((ecode = read_pk(conf.pkfile, &k, &P, &Nk))) {
    P_EXT("please check your input linear power spectrum.\n");
    return ecode;
  }
  if ((ecode = init_pk(k, P, Nk, &conf))) {
    P_EXT("please check your input linear power spectrum.\n");
    return ecode;
  }
  printf(FMT_DONE);

  printf("Generating the density field ... ");
  fflush(stdout);
  if ((ecode = init_field(conf.Ngrid, &mesh, &fp))) {
    P_EXT("failed to generate the density field.\n");
    return ecode;
  }
  if ((ecode = gauss_ran_field(&conf, k, P, Nk, &fp, mesh))) {
    P_EXT("failed to generate the density field.\n");
    return ecode;
  }
  free(k);
  free(P);
  printf(FMT_DONE);

  printf("Populating haloes ... ");
  select_dens(mesh, conf.Ngrid, conf.lowNg, conf.Nhalo);
  fflush(stdout);
  printf(FMT_DONE);

  printf("Saving haloes ... ");
  fflush(stdout);
  if ((ecode = save_halo(conf.output, mesh, conf.Ngrid, conf.Lbox))) {
    P_EXT("failed to generate the halo catalogue.\n");
    return ecode;
  }
  printf(FMT_DONE);

  if (conf.savedm) {
    printf("Saving the density field ... ");
    fflush(stdout);
    if ((ecode = save_dens(conf.dmout, mesh, conf.lowNg))) {
      P_EXT("failed to save the density field.\n");
      return ecode;
    }
    printf(FMT_DONE);
  }

#ifdef DOUBLE_PREC
  fftw_free(mesh);
#else
  fftwf_free(mesh);
#endif
  return 0;
}

