#!/bin/bash

#PBS -q GPU-1A
#PBS -N hifidiffv18r9
#PBS -l select=1:ngpus=1
#PBS -j oe
#PBS -M s2212015@jaist.ac.jp -m be

module load singularity
cd ~/HifiDiff
singularity exec -i --nv ~/pytorch_22.02-py3.sif python __main__.py \
    configs/hifidiffv18r9.py \
    checkpoints/hifidiffv18r9 \
    /home/s2212015/LJSpeech-1.1 \
    filelists/train.txt \
    --max_steps 1000000 --validate_loop 10000 --save_ckpt_loop 50000 >> logs/hifidiffv18r9.log 2>&1