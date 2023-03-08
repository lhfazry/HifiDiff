#!/bin/bash

#PBS -q GPU-1
#PBS -N infereval_hifidiffv18r19
#PBS -l select=1:ngpus=1
#PBS -j oe
#PBS -M s2212015@jaist.ac.jp -m be

module load singularity
cd ~/HifiDiff
singularity exec -i --nv ~/pytorch_22.02-py3.sif python inference.py \
    checkpoints/hifidiffv18r19 \
    /home/s2212015/LJSpeech-1.1 \
    filelists/all_test_kagayaki.txt \
    --step 100000 \
    --fast_iter 50 >> logs/inferences/hifidiffv18r19.log 2>&1

singularity exec -i --nv ~/pytorch_22.02-py3.sif python tools/evaluate2.py \
    --prefix LJ \
    --sdir checkpoints/sample_slow/hifidiffv18r19_step100000 \
    --odir /home/s2212015/LJSpeech-1.1/wavs >> logs/evaluation/hifidiffv18r19_step100000.log 2>&1

singularity exec -i --nv ~/pytorch_22.02-py3.sif python tools/evaluate2.py \
    --prefix Ind001_F_Bli \
    --sdir checkpoints/sample_slow/hifidiffv18r19_step100000 \
    --odir /home/s2212015/indsp/Ind001_F_Bli/ >> logs/evaluation/hifidiffv18r19_step100000.log 2>&1

singularity exec -i --nv ~/pytorch_22.02-py3.sif python tools/evaluate2.py \
    --prefix Ind002_M_Bli \
    --sdir checkpoints/sample_slow/hifidiffv18r19_step100000 \
    --odir /home/s2212015/indsp/Ind002_M_Bli/ >> logs/evaluation/hifidiffv18r19_step100000.log 2>&1

singularity exec -i --nv ~/pytorch_22.02-py3.sif python tools/evaluate2.py \
    --prefix Ind001_F_Btk \
    --sdir checkpoints/sample_slow/hifidiffv18r19_step100000 \
    --odir /home/s2212015/indsp/Ind001_F_Btk/ >> logs/evaluation/hifidiffv18r19_step100000.log 2>&1

singularity exec -i --nv ~/pytorch_22.02-py3.sif python tools/evaluate2.py \
    --prefix Ind002_M_Btk \
    --sdir checkpoints/sample_slow/hifidiffv18r19_step100000 \
    --odir /home/s2212015/indsp/Ind002_M_Btk/ >> logs/evaluation/hifidiffv18r19_step100000.log 2>&1

singularity exec -i --nv ~/pytorch_22.02-py3.sif python tools/evaluate2.py \
    --prefix Ind001_F_Jaw \
    --sdir checkpoints/sample_slow/hifidiffv18r19_step100000 \
    --odir /home/s2212015/indsp/Ind001_F_Jaw/ >> logs/evaluation/hifidiffv18r19_step100000.log 2>&1

singularity exec -i --nv ~/pytorch_22.02-py3.sif python tools/evaluate2.py \
    --prefix Ind002_M_Jaw \
    --sdir checkpoints/sample_slow/hifidiffv18r19_step100000 \
    --odir /home/s2212015/indsp/Ind002_M_Jaw/ >> logs/evaluation/hifidiffv18r19_step100000.log 2>&1

singularity exec -i --nv ~/pytorch_22.02-py3.sif python tools/evaluate2.py \
    --prefix Ind001_F_Snd \
    --sdir checkpoints/sample_slow/hifidiffv18r19_step100000 \
    --odir /home/s2212015/indsp/Ind001_F_Snd/ >> logs/evaluation/hifidiffv18r19_step100000.log 2>&1

singularity exec -i --nv ~/pytorch_22.02-py3.sif python tools/evaluate2.py \
    --prefix Ind002_M_Snd \
    --sdir checkpoints/sample_slow/hifidiffv18r19_step100000 \
    --odir /home/s2212015/indsp/Ind002_M_Snd/ >> logs/evaluation/hifidiffv18r19_step100000.log 2>&1

singularity exec -i --nv ~/pytorch_22.02-py3.sif python tools/evaluate2.py \
    --prefix afr_ \
    --sdir checkpoints/sample_slow/hifidiffv18r19_step100000 \
    --odir /home/s2212015/datasets/af_za/za/afr/wavs/ >> logs/evaluation/hifidiffv18r19_step100000.log 2>&1

singularity exec -i --nv ~/pytorch_22.02-py3.sif python tools/evaluate2.py \
    --prefix arf_ \
    --sdir checkpoints/sample_slow/hifidiffv18r19_step100000 \
    --odir /home/s2212015/datasets/es_ar_female/ >> logs/evaluation/hifidiffv18r19_step100000.log 2>&1

singularity exec -i --nv ~/pytorch_22.02-py3.sif python tools/evaluate2.py \
    --prefix sso_ \
    --sdir checkpoints/sample_slow/hifidiffv18r19_step100000 \
    --odir /home/s2212015/datasets/st_za/za/sso/wavs/ >> logs/evaluation/hifidiffv18r19_step100000.log 2>&1

singularity exec -i --nv ~/pytorch_22.02-py3.sif python tools/evaluate2.py \
    --prefix tsn_ \
    --sdir checkpoints/sample_slow/hifidiffv18r19_step100000 \
    --odir /home/s2212015/datasets/tn_za/za/tsn/wavs/ >> logs/evaluation/hifidiffv18r19_step100000.log 2>&1

singularity exec -i --nv ~/pytorch_22.02-py3.sif python tools/evaluate2.py \
    --prefix xho_ \
    --sdir checkpoints/sample_slow/hifidiffv18r19_step100000 \
    --odir /home/s2212015/datasets/xh_za/za/xho/wavs/ >> logs/evaluation/hifidiffv18r19_step100000.log 2>&1
