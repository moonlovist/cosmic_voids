#include "rmol.h"

#define PI 3.1415926535

int save_res(char fname[MAX_BUF], long N) {
  FILE *fps;
  long i;
  double vol;

  printf("Start writing to file `%s' ...\n", fname);
  if(!(fps = fopen(fname, "w"))) {
    fprintf(stderr, "Error: cannot write in file `%s'.\n", fname);
    return -1;
  }

  for(i = 0; i < N; i++) {
    if(voids[i].x < 0 || voids[i].y < 0 || voids[i].z < 0 ||
        voids[i].x >= 2500 || voids[i].y >= 2500 || voids[i].z >= 2500)
      continue;
    if(voids[i].cell == -1) {
      vol = voids[i].r * voids[i].r * voids[i].r * 4 * PI / 3.0;
      fprintf(fps, "%.12g %.12g %.12g %.12g %.12g\n", voids[i].x, voids[i].y, voids[i].z, voids[i].r, vol);
    }
  }

  fclose(fps);
  printf("Done with writing.\n\n");
  free(voids);
  return 0;
}
