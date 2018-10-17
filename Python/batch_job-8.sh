#!/bin/bash

#PBS -l walltime=100:00:00,nodes=1:ppn=16,mem=32gb 
#PBS -m abe 
#PBS -M sd.lobo251@uniandes.edu.co
#PBS -N krr_grid_search8

#Falta correr en el archivo original sin -test1

cd $PWD
TEMP_DIR=/state/partition1/$USER/$PBS_JOBNAME.$PBS_JOBID
OUT_DIR=$PBS_O_WORKDIR/results

mkdir -p $TEMP_DIR
mkdir -p $OUT_DIR
cp -Rf $PBS_O_WORKDIR/krr_grid_search-test1.py $TEMP_DIR/.

module load anaconda/python3

cd $TEMP_DIR
nohup python krr_grid_search-test1.py --n_jobs=8 > krr_njobs8.txt

cd $OUT_DIR
mv -f $TEMP_DIR ./
