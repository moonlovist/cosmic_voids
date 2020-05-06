CC = gcc
LIBS = 
OPTS = -g -Wall $(INCL) $(LIBS) -O3
OBJS = read_conf.o read_cat.o save_res.o rmol.o
EXEC = rmol

all: $(OBJS)
	$(CC) $(OBJS) -o $(EXEC) $(OPTS)

read_conf.o: read_conf.c read_conf.h
	$(CC) -c read_conf.c $(OPTS)

read_cat.o: read_cat.c rmol.h
	$(CC) -c read_cat.c $(OPTS)

save_res.o: save_res.c rmol.h
	$(CC) -c save_res.c $(OPTS)

rmol.o: rmol.c read_conf.h rmol.h
	$(CC) -c rmol.c $(OPTS)

clean:
	rm -f $(OBJS) $(EXEC)
