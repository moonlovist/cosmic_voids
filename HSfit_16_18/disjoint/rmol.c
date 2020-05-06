#include <unistd.h>
#include "rmol.h"

int main(int argc, char *argv[]) {
  char opts;
  char *conf_file = DEFAULT_CONF_FILE;

  init_conf();

  while((opts = getopt(argc, argv, "hdc:i:n:l:r:g:o:")) != -1) {
    switch(opts) {
    case 'h':
      usage(argv[0]);
      return 0;
      break;
    case 'd':
      temp_conf();
      return 0;
      break;
    case 'c':
      conf_file = optarg;
      break;
    case 'i':
      strcpy(conf.catalog, optarg);
      break;
    case 'n':
      conf.header = atoi(optarg);
      break;
    case 'l':
      conf.bmin = atof(optarg);
    case 'r':
      conf.bmax = atof(optarg);
      break;
    case 'g':
      conf.Ngrid = atoi(optarg);
      break;
    case 'o':
      strcpy(conf.output, optarg);
      break;
    default:
      break;
    }
  }

  read_conf(conf_file);
  if(check_conf()) {
    fprintf(stderr, "Exit: please check your configuration.\n \
Try the -h option for more information.\n");
    return -1;
  }

  if(read_void(conf.catalog, conf.header, conf.bmin, conf.bmax)) {
    fprintf(stderr, "Exit: failed to read the halo catalog.\n");
    return -2;
  }

  if(rmol(conf.Ngrid, conf.bmin, conf.boxsize)) {
    fprintf(stderr, "Exit: failed to remove overlapping voids.\n");
    return -3;
  }

  if(save_res(conf.output, Nv)) {
    fprintf(stderr, "Exit: failed to write power spectrum.\n");
    return -4;
  }

  printf("All done.\n");
  return 0;
}


int rmol(int Ngrid, double bmin, double bs) {
  long i, j, k, cell;
  long **v_in_cell;
  long *Nv_in_cell;
  int a, b, c, xmin, xmax, ymin, ymax, zmin, zmax;
  double rsum, dx, dy, dz;

  printf("Sorting voids according to the radius ... ");
  fflush(stdout);
  qsort(voids, Nv, sizeof(struct voidst), compare_radius);
  printf("Done.\nConnecting voids with cells ...\n Allocating memory ... ");
  fflush(stdout);

  v_in_cell = (long **) malloc(sizeof(long *) * GNUM);
  Nv_in_cell = (long *) malloc(sizeof(long) * GNUM);
  if(!v_in_cell || !Nv_in_cell) {
    fprintf(stderr, "Error: failed to allocate memory.\n");
    return -1;
  }
  memset(Nv_in_cell, 0, sizeof(long) * GNUM);
  printf("Done with %lg MB.\n Counting number of voids in each cell ... ",
      (sizeof(long *) + sizeof(long)) * GNUM / (double) (1024 * 1024));
  fflush(stdout);

  for(i = 0; i < Nv; i++) {
    a = (voids[i].x - bmin) * Ngrid / bs;
    b = (voids[i].y - bmin) * Ngrid / bs;
    c = (voids[i].z - bmin) * Ngrid / bs;
    if(a < 0 || b < 0 || c < 0 || a >= Ngrid || b >= Ngrid || c >= Ngrid) {
      fprintf(stderr, "Error: void (%.12g %.12g %.12g %.12g) is outside the boundary.\n", voids[i].x, voids[i].y, voids[i].z, voids[i].r);
      return -3;
    }
    voids[i].cell = (long) a * Ngrid * Ngrid + b * Ngrid + c;
    Nv_in_cell[voids[i].cell]++;
  }
  printf("Done.\n Recording voids in each cell ... ");
  fflush(stdout);

  for(i = 0; i < GNUM; i++) {
    if(Nv_in_cell[i]) {
      v_in_cell[i] = (long *) malloc(sizeof(long) * Nv_in_cell[i]);
      if(!v_in_cell[i]) {
        fprintf(stderr, "Error: failed to allocate memory.\n");
        return -2;
      }
    }
    else
      v_in_cell[i] = NULL;
  }

  memset(Nv_in_cell, 0, sizeof(long) * GNUM);
  for(i = 0; i < Nv; i++) {
    v_in_cell[voids[i].cell][Nv_in_cell[voids[i].cell]] = i;
    Nv_in_cell[voids[i].cell]++;
  }
  printf("Done.\nDone.\n\nRemoving overlapping voids ... ");
  fflush(stdout);

  for(i = 0; i < Nv; i++) {
    if(voids[i].cell < 0) continue;
    voids[i].cell = -1;
    xmin = (voids[i].x - 2 * voids[i].r - bmin) * Ngrid / bs;
    xmax = (voids[i].x + 2 * voids[i].r - bmin) * Ngrid / bs;
    ymin = (voids[i].y - 2 * voids[i].r - bmin) * Ngrid / bs;
    ymax = (voids[i].y + 2 * voids[i].r - bmin) * Ngrid / bs;
    zmin = (voids[i].z - 2 * voids[i].r - bmin) * Ngrid / bs;
    zmax = (voids[i].z + 2 * voids[i].r - bmin) * Ngrid / bs;

    if(xmin < 0) xmin = 0;
    if(ymin < 0) ymin = 0;
    if(zmin < 0) zmin = 0;
    if(xmax >= Ngrid) xmax = Ngrid - 1;
    if(ymax >= Ngrid) ymax = Ngrid - 1;
    if(zmax >= Ngrid) zmax = Ngrid - 1;

    for(a = xmin; a <= xmax; a++)
      for(b = ymin; b <= ymax; b++)
        for(c = zmin; c <= zmax; c++) {
          cell = (long) a * Ngrid * Ngrid + b * Ngrid + c;
          if(Nv_in_cell[cell] == 0) continue;

          for(k = 0; k < Nv_in_cell[cell]; k++) {
            j = v_in_cell[cell][k];
            if(voids[j].cell < 0) continue;
            rsum = voids[i].r + voids[j].r;
            dx = voids[i].x - voids[j].x;
            dy = voids[i].y - voids[j].y;
            dz = voids[i].z - voids[j].z;
            if(dx > rsum || dx < -rsum || dy > rsum || dy < -rsum ||
                dz > rsum || dz < -rsum)
              continue;
            if(dx * dx + dy * dy + dz * dz < rsum * rsum) {
//              Nv--;
//              voids[j] = voids[Nv];
              voids[j].cell = -2;
              Nv_in_cell[cell]--;
              v_in_cell[cell][k] = v_in_cell[cell][Nv_in_cell[cell]];
              k--;
            }
          }
        }
  }

//  printf("Done.\nSorting voids according to the radius ... ");
//  fflush(stdout);
//  qsort(voids, Nv, sizeof(struct voidst), compare_radius);
  printf("Done.\nReleasing memory ... ");
  fflush(stdout);
  free(Nv_in_cell);
  for(i = 0; i < GNUM; i++)
    if(v_in_cell[i] != NULL) free(v_in_cell[i]);
  free(v_in_cell);
  printf("Done.\n\n");

  return 0;
}


int compare_radius(const void *a, const void *b) {
  if(((struct voidst *)a)->r < ((struct voidst *)b)->r) return +1;
  if(((struct voidst *)a)->r > ((struct voidst *)b)->r) return -1;
  return 0;
}


void usage(char *pname) {
  printf(" Usage: %s [OPTION [VALUE]]\n\
    Assign mass to haloes.\n\n\
      -c  Configuration file.\n\
          DEFAULT: `%s'.\n\
      -d  Display the default configuration and exit.\n\
      -p  Input dark matter density field (binary).\n\
      -w  Path of input cosmic web files.\n\
      -a  Input halo catalog.\n\
      -i  Input bias information.\n\
      -l  Number of header lines for the halo catalog.\n\
          DEFAULT: 0.\n\
      -b  Box size of the catalogs, in Mpc/h.\n\
      -g  Grid size of the input DM density field.\n\
      -s  Random seed (positive integer).\n\
      -t  Mass threshold for the assignment.\n\
      -o  The output file.\n\
      -h  Display this message and exit.\n", pname, DEFAULT_CONF_FILE);
}
