#!/bin/bash

#PBS -q GPU-1A
#PBS -N infer_diffwave
#PBS -l select=1:ngpus=1
#PBS -j oe
#PBS -M s2212015@jaist.ac.jp -m be

module load singularity
cd ~/HifiDiff
singularity exec -i --nv ~/pytorch_22.02-py3.sif python inference.py \
    checkpoints/diffwave \
    /home/s2212015/LJSpeech-1.1 \
    filelists/test.txt \
    --step 1000000 \
    --fast_iter 50 >> logs/inferences/diffwave.log 2>&1
singularity exec -i --nv ~/pytorch_22.02-py3.sif python inference.py \
    checkpoints/diffwave \
    /home/s2212015/indsp \
    filelists/test_bali.txt \
    --step 1000000 \
    --fast_iter 50 >> logs/inferences/diffwave.log 2>&1
singularity exec -i --nv ~/pytorch_22.02-py3.sif python inference.py \
    checkpoints/diffwave \
    /home/s2212015/indsp \
    filelists/test_batak.txt \
    --step 1000000 \
    --fast_iter 50 >> logs/inferences/diffwave.log 2>&1
singularity exec -i --nv ~/pytorch_22.02-py3.sif python inference.py \
    checkpoints/diffwave \
    /home/s2212015/indsp \
    filelists/test_jawa.txt \
    --step 1000000 \
    --fast_iter 50 >> logs/inferences/diffwave.log 2>&1
singularity exec -i --nv ~/pytorch_22.02-py3.sif python inference.py \
    checkpoints/diffwave \
    /home/s2212015/indsp \
    filelists/test_sunda.txt \
    --step 1000000 \
    --fast_iter 50 >> logs/inferences/diffwave.log 2>&1
    