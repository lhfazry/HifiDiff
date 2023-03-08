#!/bin/bash

#PBS -q GPU-1
#PBS -N infer_hifidiffv18r17
#PBS -l select=1:ngpus=1
#PBS -j oe
#PBS -M s2212015@jaist.ac.jp -m be

module load singularity
cd ~/HifiDiff
singularity exec -i --nv ~/pytorch_22.02-py3.sif python inference.py \
    checkpoints/hifidiffv18r17 \
    /home/s2212015/LJSpeech-1.1 \
    filelists/all_test_kagayaki.txt \
    --step 50000 \
    --fast_iter 50 >> logs/inferences/hifidiffv18r17.log 2>&1