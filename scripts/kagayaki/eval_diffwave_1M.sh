#!/bin/bash

#PBS -q GPU-1A
#PBS -N eval_diffwave_1M
#PBS -l select=1:ngpus=1
#PBS -j oe
#PBS -M s2212015@jaist.ac.jp -m be

module load singularity
cd ~/HifiDiff
singularity exec -i --nv ~/pytorch_22.02-py3.sif python tools/evaluate2.py \
    --prefix LJ \
    --sdir checkpoints/sample_slow/diffwave_step1000000 \
    --odir /home/s2212015/LJSpeech-1.1/wavs >> logs/evaluation/diffwave_step1000000.log 2>&1

singularity exec -i --nv ~/pytorch_22.02-py3.sif python tools/evaluate2.py \
    --prefix Ind001_F_Bli \
    --sdir checkpoints/sample_slow/diffwave_step1000000 \
    --odir /home/s2212015/indsp/Ind001_F_Bli/ >> logs/evaluation/diffwave_step1000000.log 2>&1

singularity exec -i --nv ~/pytorch_22.02-py3.sif python tools/evaluate2.py \
    --prefix Ind002_M_Bli \
    --sdir checkpoints/sample_slow/diffwave_step1000000 \
    --odir /home/s2212015/indsp/Ind002_M_Bli/ >> logs/evaluation/diffwave_step1000000.log 2>&1

singularity exec -i --nv ~/pytorch_22.02-py3.sif python tools/evaluate2.py \
    --prefix Ind001_F_Btk \
    --sdir checkpoints/sample_slow/diffwave_step1000000 \
    --odir /home/s2212015/indsp/Ind001_F_Btk/ >> logs/evaluation/diffwave_step1000000.log 2>&1

singularity exec -i --nv ~/pytorch_22.02-py3.sif python tools/evaluate2.py \
    --prefix Ind002_M_Btk \
    --sdir checkpoints/sample_slow/diffwave_step1000000 \
    --odir /home/s2212015/indsp/Ind002_M_Btk/ >> logs/evaluation/diffwave_step1000000.log 2>&1

singularity exec -i --nv ~/pytorch_22.02-py3.sif python tools/evaluate2.py \
    --prefix Ind001_F_Jaw \
    --sdir checkpoints/sample_slow/diffwave_step1000000 \
    --odir /home/s2212015/indsp/Ind001_F_Jaw/ >> logs/evaluation/diffwave_step1000000.log 2>&1

singularity exec -i --nv ~/pytorch_22.02-py3.sif python tools/evaluate2.py \
    --prefix Ind002_M_Jaw \
    --sdir checkpoints/sample_slow/diffwave_step1000000 \
    --odir /home/s2212015/indsp/Ind002_M_Jaw/ >> logs/evaluation/diffwave_step1000000.log 2>&1

singularity exec -i --nv ~/pytorch_22.02-py3.sif python tools/evaluate2.py \
    --prefix Ind001_F_Snd \
    --sdir checkpoints/sample_slow/diffwave_step1000000 \
    --odir /home/s2212015/indsp/Ind001_F_Snd/ >> logs/evaluation/diffwave_step1000000.log 2>&1

singularity exec -i --nv ~/pytorch_22.02-py3.sif python tools/evaluate2.py \
    --prefix Ind002_M_Snd \
    --sdir checkpoints/sample_slow/diffwave_step1000000 \
    --odir /home/s2212015/indsp/Ind002_M_Snd/ >> logs/evaluation/diffwave_step1000000.log 2>&1

singularity exec -i --nv ~/pytorch_22.02-py3.sif python tools/evaluate2.py \
    --prefix afr_ \
    --sdir checkpoints/sample_slow/diffwave_step1000000 \
    --odir /home/s2212015/datasets/af_za/za/afr/wavs/ >> logs/evaluation/diffwave_step1000000.log 2>&1

singularity exec -i --nv ~/pytorch_22.02-py3.sif python tools/evaluate2.py \
    --prefix arf_ \
    --sdir checkpoints/sample_slow/diffwave_step1000000 \
    --odir /home/s2212015/datasets/es_ar_female/ >> logs/evaluation/diffwave_step1000000.log 2>&1

singularity exec -i --nv ~/pytorch_22.02-py3.sif python tools/evaluate2.py \
    --prefix sso_ \
    --sdir checkpoints/sample_slow/diffwave_step1000000 \
    --odir /home/s2212015/datasets/st_za/za/sso/wavs/ >> logs/evaluation/diffwave_step1000000.log 2>&1

singularity exec -i --nv ~/pytorch_22.02-py3.sif python tools/evaluate2.py \
    --prefix tsn_ \
    --sdir checkpoints/sample_slow/diffwave_step1000000 \
    --odir /home/s2212015/datasets/tn_za/za/tsn/wavs/ >> logs/evaluation/diffwave_step1000000.log 2>&1

singularity exec -i --nv ~/pytorch_22.02-py3.sif python tools/evaluate2.py \
    --prefix xho_ \
    --sdir checkpoints/sample_slow/diffwave_step1000000 \
    --odir /home/s2212015/datasets/xh_za/za/xho/wavs/ >> logs/evaluation/diffwave_step1000000.log 2>&1

singularity exec -i --nv ~/pytorch_22.02-py3.sif python tools/evaluate2.py \
    --prefix fr_ \
    --sdir checkpoints/sample_slow/diffwave_step1000000 \
    --odir /home/s2212015/datasets/xh_za/za/xho/wavs/ >> logs/evaluation/diffwave_step1000000.log 2>&1
