#!/bin/bash
#
## Number of cores per task
#
##export NUM_THREADSPROCESSES=${SLURM_CPUS_PER_TASK}#
export OMP_NUM_THREADS=16
let number=$((25*100))
echo $number
sed -i -e "s/replace/$number/g" /home/epfl/tan/first-code-document/linhalo/linhalo.conf
srun --mpi=pmi2 ./linhalo
sed -i -e "s/$number/replace/g" /home/epfl/tan/first-code-document/linhalo/linhalo.conf

echo All done


