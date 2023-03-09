#!/bin/bash

#PBS -q GPU-1
#PBS -N eval_diffwave
#PBS -l select=1:ngpus=1
#PBS -j oe
#PBS -M s2212015@jaist.ac.jp -m be

module load singularity
cd ~/HifiDiff
singularity exec -i --nv ~/pytorch_22.02-py3.sif python tools/evaluate2.py \
    --prefix fr_ \
    --sdir checkpoints/sample_slow/diffwave_step300000 \
    --odir /home/s2212015/datasets/fr/CLEM_HDD/IRCAM/Open_SLR/wav/ >> logs/evaluation/diffwave.log 2>&1
