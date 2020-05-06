/*
   Created by Cheng Zhao on 6-Jan-2014
*/

#ifndef READ_CONF_H
#define READ_CONF_H

#define DEFAULT_CONF_FILE "rmol.conf"
#define DEFAULT_HEADER 0
#define MAX_BUF 255
#define TIMEOUT 20

struct config {
  char catalog[MAX_BUF];
  int header;
  double bmin;
  double bmax;
  double boxsize;
  int Ngrid;
  char output[MAX_BUF];
} conf;

void init_conf(void);

int read_conf(char*);

int check_conf(void);

void temp_conf(void);

#endif
