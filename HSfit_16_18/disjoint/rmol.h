#ifndef RMOL_H
#define RMOL_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <malloc.h>
#include "read_conf.h"

#define GNUM (long)Ngrid*Ngrid*Ngrid

struct voidst {
  double x;
  double y;
  double z;
  double r;
  long cell;
} *voids;

long Nv; //number of voids.

int read_void(char [MAX_BUF], int, double, double);

int rmol(int, double, double);

int save_res(char [MAX_BUF], long);

int compare_radius(const void *, const void *);

void usage(char *);

#endif
