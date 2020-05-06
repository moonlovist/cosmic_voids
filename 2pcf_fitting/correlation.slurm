#!/bin/bash
# Number of cores per task
##export NUM_THREADSPROCESSES=${SLURM_CPUS_PER_TASK}#

export OMP_NUM_THREADS=16
for counter in {2..100}
do
  let number=$(($counter*100))
  echo $number
  sed -i -e "s/replace/$number/g" /hpcstorage/zhaoc/share/ting/2PCF_disjoint_voids/cf_periodic.py
  cd pair_counts_auto_disjoint_16
  /hpcstorage/zhaoc/share/ting/2PCF_disjoint_voids/cf_periodic.py DR_pair_counts_$number.dat 2500 
  sed -i -e "s/$number/replace/g" /hpcstorage/zhaoc/share/ting/2PCF_disjoint_voids/cf_periodic.py
  cd ..
done
echo All done

