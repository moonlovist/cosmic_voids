#!/bin/bash
# Number of cores per task
##export NUM_THREADSPROCESSES=${SLURM_CPUS_PER_TASK}#

export OMP_NUM_THREADS=16
srun --mpi=pmi2 python3 HSfit.py

