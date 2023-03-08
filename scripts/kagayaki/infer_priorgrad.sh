#!/bin/bash

#PBS -q GPU-1A
#PBS -N infer_hifidiffv18r4_mstft
#PBS -l select=1:ngpus=1
#PBS -j oe
#PBS -M s2212015@jaist.ac.jp -m be

module load singularity
cd ~/HifiDiff
singularity exec -i --nv ~/pytorch_22.02-py3.sif python inference.py \
    checkpoints/priorgrad \
    /home/s2212015/LJSpeech-1.1 \
    filelists/test.txt \
    --step 550000 \
    --fast_iter 50 >> logs/inferences/priorgrad.log 2>&1
singularity exec -i --nv ~/pytorch_22.02-py3.sif python inference.py \
    checkpoints/priorgrad \
    /home/s2212015/indsp \
    filelists/test_bali.txt \
    --step 550000 \
    --fast_iter 50 >> logs/inferences/priorgrad.log 2>&1
singularity exec -i --nv ~/pytorch_22.02-py3.sif python inference.py \
    checkpoints/priorgrad \
    /home/s2212015/indsp \
    filelists/test_batak.txt \
    --step 550000 \
    --fast_iter 50 >> logs/inferences/priorgrad.log 2>&1
singularity exec -i --nv ~/pytorch_22.02-py3.sif python inference.py \
    checkpoints/priorgrad \
    /home/s2212015/indsp \
    filelists/test_jawa.txt \
    --step 550000 \
    --fast_iter 50 >> logs/inferences/priorgrad.log 2>&1
singularity exec -i --nv ~/pytorch_22.02-py3.sif python inference.py \
    checkpoints/priorgrad \
    /home/s2212015/indsp \
    filelists/test_sunda.txt \
    --step 550000 \
    --fast_iter 50 >> logs/inferences/hifidiffv18r4_mstft.log 2>&1
    