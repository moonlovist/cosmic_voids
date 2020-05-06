#include "rmol.h"

#define BUF 512
#define CHUNK 1048576

int read_void(char fname[MAX_BUF], int header, double min, double max) {
  FILE *fcat;
  char line[BUF];
  char *buffer, *p, *end;
  size_t count;
  long i, bytes, flag[21];
  double x, y, z, r;
  int j;

  printf("Counting lines of file `%s' ... ", fname);
  fflush(stdout);

  Nv = 0;
  buffer = (char *) malloc(sizeof(char) * CHUNK);
  if(!buffer) {
    fprintf(stderr, "\nError: failed to allocate memory for reading the file.\n");
    return -2;
  }

  if(!(fcat = fopen(fname, "rb"))) {
   fprintf(stderr, "\nError: failed to open file `%s'.\n", fname);
   return -1;
  }
  while(!feof(fcat)) {
    if((count = fread(buffer, sizeof(char), CHUNK, fcat)) < 0) {
      fprintf(stderr, "\nError: failed to read file `%s'.\n", fname);
      return -1;
    }
    p = buffer;
    end = p + count;
    while((p = memchr(p, '\n', end - p))) {
      ++p;
      ++Nv;
    }
  }
  fclose(fcat);
  free(buffer);
  Nv -= header;
  printf("Done.\nNumber of records: %ld\n", Nv);

  //Allocate memory for the particles.
  voids = (struct voidst *) malloc(bytes = sizeof(struct voidst) * Nv);
  if(!voids) {
    fprintf(stderr, "Error: failed to allocate %g Mb for the voids.\n", bytes / (1024.0 * 1024.0));
    return -2;
  }
  printf("Allocated %g Mb for the voids.\n\n", bytes / (1024.0 * 1024.0));

//Open and read catalog.
  if(!(fcat = fopen(fname, "r"))) {
    fprintf(stderr, "Error: cannot open catalog file `%s'.\n", fname);
    return -1;
  }
  printf("Start reading catalog `%s' ...\n", fname);

//Skip header.
  for(i = 0; i < header; i++)
    if(!(fgets(line, BUF, fcat)))
      fprintf(stderr, "Warning: line %lu of the header may not be correct.\n", i);

//Read data.
  for(j = 0; j < 21; j++)
    flag[j] = Nv * j * 5 / 100;
  printf("Reading ...  0%%");
  fflush(stdout);
  for(bytes = 0, count = 0, j = 1; j < 21; j++) {
    for(i = flag[j - 1]; i < flag[j]; i++) {
      if(!(fgets(line, BUF, fcat))) {
        fprintf(stderr, "\nError: line %lu is not correct, reading aborted.\n", i + header + 2);
        return -3;
      }

      sscanf(line, "%lf %lf %lf %lf", &x, &y, &z, &r);
      if(x < min || y < min || z < min || x >= max || y >= max || z >= max) {
//        fprintf(stderr, "Warning: throwing void (%.12g %.12g %.12g %.12g) outside the boundary.\n", x, y, z, r);
        count++;
        continue;
      }
      voids[bytes].x = x;
      voids[bytes].y = y;
      voids[bytes].z = z;
      voids[bytes].r = r;
      bytes++;
    }
    if(j != 20) {
      printf("\b\b\b\b%3d%%", j * 5);
      fflush(stdout);
    }
  }
  if(bytes + count != Nv) {
    fprintf(stderr, "Warning: number of voids (%ld) and records in the file do not match.\n", bytes + count);
  }
  Nv = bytes;
  printf("\b\b\bDone with %ld voids\nRemoved %ld voids outside the boundary.\n\n", Nv, count);

  if(Nv < 2) {
    fprintf(stderr, "Error: not enough voids.\n");
    return -5;
  }

  fclose(fcat);
  return 0;
}

