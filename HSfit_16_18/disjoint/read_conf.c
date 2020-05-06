#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <unistd.h>
#include "read_conf.h"

void init_conf(void) {
  memset(conf.catalog, '\0', MAX_BUF);
  memset(conf.output, '\0', MAX_BUF);
  conf.header = -1;
  conf.bmin = 1e99;
  conf.bmax = -1e99;
  conf.boxsize = -1;
  conf.Ngrid = -1;
}


int read_conf(char *fname) {
  FILE *fconf;
  char line[MAX_BUF + 15];
  char keyword[10];
  char value[MAX_BUF];

  if(!(fconf = fopen(fname, "r"))) {
    fprintf(stderr, "Warning: cannot open configuration file `%s'.\n", fname);
    return -1;
  }
  while(fgets(line, MAX_BUF + 15, fconf) != NULL) {
    memset(value, '\0', MAX_BUF);
    sscanf(line, "%*[ |\t]%[^\n]", line);       //remove first whitespaces
    if(line[0] == '#') continue;                //ignore comments
    sscanf(line, "%[^=]=%[^={#|\n}]", keyword, value);

    sscanf(keyword, "%s", keyword);             //remove whitespaces
    sscanf(value, "%*[ |\t]%[^\n]", value);
    if(value[0] == '\"') {                      //deal with quotation marks
      sscanf(value, "%*c%[^\"]", value);
    } else {
      sscanf(value, "%s", value);
    }

    if(!strcmp(keyword, "CATALOG") && conf.catalog[0] == '\0')
      strcpy(conf.catalog, value);
    else if(!strcmp(keyword, "OUTPUT") && conf.output[0] == '\0')
      strcpy(conf.output, value);
    else if(!strcmp(keyword, "HEADER") && conf.header == -1)
      conf.header = atoi(value);
    else if(!strcmp(keyword, "BOXMIN") && conf.bmin > 1e98)
      conf.bmin = atof(value);
    else if(!strcmp(keyword, "BOXMAX") && conf.bmax < -1e98)
      conf.bmax = atof(value);
    else if(!strcmp(keyword, "NGRID") && conf.Ngrid == -1)
      conf.Ngrid = atoi(value);
    else continue;
  }

  fclose(fconf);
  return 0;
}

int check_conf(void) {
  FILE *fps;
  char confirm;
  int exit = 0;

//Check if the parameters are set.
  if(conf.catalog[0] == '\0' || conf.catalog[0] == ' ') {
    fprintf(stderr, "Error: the input void catalog is not set.\n");
    return -1;
  }
  conf.boxsize = conf.bmax - conf.bmin;
  if(conf.boxsize <= 0) {
    fprintf(stderr, "Error: the box size is not set correctly.\n");
    return -2;
  }
  if(conf.Ngrid < 1) {
    fprintf(stderr, "Error: the grid size is not set correctly.\n");
    return -3;
  }
  if(conf.output[0] == '\0' || conf.output[0] == ' ') {
    fprintf(stderr, "Error: the output file is not set.\n");
    return -4;
  }

  if(conf.header < 0) {
    fprintf(stderr, "Warning: number of header lines is not set correctly.\n\
Employ the dedault value (%d) instead.", DEFAULT_HEADER);
    conf.header = DEFAULT_HEADER;
  }

  if(access(conf.catalog, R_OK)) {
    fprintf(stderr, "Error: cannot open the density field catalog: `%s'.\n", conf.catalog);
    return -1;
  }

  if(!access(conf.output, F_OK)) {
    fprintf(stderr, "Warning: the output file `%s' is exist.\n", conf.output);
    do {
      if((++exit) == TIMEOUT) {
        fprintf(stderr, "\nError: too many failed inputs.\n");
        return -7;
      }
      fprintf(stderr, "Are you going to overwrite it? (y/n): ");
      if(scanf("%c", &confirm) != 1) continue;
      while(getchar() != '\n');
    }
    while(confirm != 'y' && confirm != 'n');
    if(confirm == 'n' || access(conf.output, W_OK)) {
      fprintf(stderr, "Error: cannot write to output file `%s'.\n", conf.output);
      return -4;
    }
  }
  if(!(fps = fopen(conf.output, "w"))) {
    fprintf(stderr, "Error: cannot write in file `%s'.\n", conf.output);
    return -4;
  }
  fclose(fps);

  printf("Configuration loaded successfully.\n\n");
  return 0;
}



void temp_conf(void) {
  printf("# Configuration file.\n\
# NOTICE that command line options have priority over this file.\n\
# format: keyword = value # comment\n\
# use double quotation marks (\") if there is whitespace (blanks/tabs) in value\n\n\
PARTCAT = \n\
        # Dark matter density field of mocks (binary file).\n\
HALOCAT = \n\
        # Halo catalog of mocks with (x,y,z,vx,vy,vz).\n\
WEBPATH = \n\
        # Path of the cosmic web type files of mocks (4 binary files).\n\
INPUT   = \n\
        # The input bias information from N-body simulations.\n\
HEADER  = 0\n\
        # Number of header lines for the halo catalog (DEFAULT: 0).\n\
BOXSIZE = 2500.0\n\
        # Box size of the catalog, in Mpc/h.\n\
NGRID   = 1024\n\
        # Grid size of the DM density field & cosmic web type files.\n\
SEED    = 441581\n\
        # Random seed for assigning mass to halos.\n\
THRESHOLD = 500\n\
        # Haloes with mass above this value will not be in the same cell.\n\
        # Set it to -1 to disable the threshold.\n\
OUTPUT  = output/test.dat\n\
        # The output file.\n");
}

