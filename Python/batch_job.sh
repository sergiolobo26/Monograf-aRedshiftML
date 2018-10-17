#!/bin/bash

#PBS -l walltime=100:00:00,nodes=1:ppn=8,mem=16gb 
#PBS -m abe 
#PBS -M sd.lobo251@uniandes.edu.co
#PBS -N krr_grid_search4

cd $PWD
TEMP_DIR=/state/partition1/$USER/$PBS_JOBNAME.$PBS_JOBID
OUT_DIR=$PBS_O_WORKDIR/results

mkdir -p $TEMP_DIR
mkdir -p $OUT_DIR
cp -Rf $PBS_O_WORKDIR/krr_grid_search-test1.py $TEMP_DIR/.

module load anaconda/python3

cd $TEMP_DIR
nohup python krr_grid_search-test1.py --n_jobs=4 > krr_njobs1.txt

cd $OUT_DIR
mv -f $TEMP_DIR ./
