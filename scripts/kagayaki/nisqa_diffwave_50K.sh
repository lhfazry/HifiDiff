#!/bin/bash

#PBS -q GPU-1
#PBS -N nisqa_diffwave
#PBS -l select=1:ngpus=1
#PBS -j oe
#PBS -M s2212015@jaist.ac.jp -m be

module load singularity
cd ~/NISQA

singularity exec -i --nv ~/pytorch_22.02-py3.sif python run_predict.py \
--mode predict_dir --prefix Ind001_F_Bli \
--pretrained_model weights/nisqa_tts.tar \
--data_dir ../HifiDiff/checkpoints/sample_slow/diffwave_step50000/ \
--num_workers 0 --bs 10 \
--output_dir results \
--filename diffwave_step50000_Ind001_F_Bli >> logs/diffwave_step50000_Ind001_F_Bli.log 2>&1

singularity exec -i --nv ~/pytorch_22.02-py3.sif python run_predict.py \
--mode predict_dir --prefix Ind002_M_Bli \
--pretrained_model weights/nisqa_tts.tar \
--data_dir ../HifiDiff/checkpoints/sample_slow/diffwave_step50000/ \
--num_workers 0 --bs 10 \
--output_dir results \
--filename diffwave_step50000_Ind002_M_Bli >> logs/diffwave_step50000_Ind002_M_Bli.log 2>&1

singularity exec -i --nv ~/pytorch_22.02-py3.sif python run_predict.py \
--mode predict_dir --prefix Ind001_F_Btk \
--pretrained_model weights/nisqa_tts.tar \
--data_dir ../HifiDiff/checkpoints/sample_slow/diffwave_step50000/ \
--num_workers 0 --bs 10 \
--output_dir results \
--filename diffwave_step50000_Ind001_F_Btk >> logs/diffwave_step50000_Ind001_F_Btk.log 2>&1

singularity exec -i --nv ~/pytorch_22.02-py3.sif python run_predict.py \
--mode predict_dir --prefix Ind002_M_Btk \
--pretrained_model weights/nisqa_tts.tar \
--data_dir ../HifiDiff/checkpoints/sample_slow/diffwave_step50000/ \
--num_workers 0 --bs 10 \
--output_dir results \
--filename diffwave_step50000_Ind002_M_Btk >> logs/diffwave_step50000_Ind002_M_Btk.log 2>&1

singularity exec -i --nv ~/pytorch_22.02-py3.sif python run_predict.py \
--mode predict_dir --prefix Ind001_F_Jaw \
--pretrained_model weights/nisqa_tts.tar \
--data_dir ../HifiDiff/checkpoints/sample_slow/diffwave_step50000/ \
--num_workers 0 --bs 10 \
--output_dir results \
--filename diffwave_step50000_Ind001_F_Jaw >> logs/diffwave_step50000_Ind001_F_Jaw.log 2>&1

singularity exec -i --nv ~/pytorch_22.02-py3.sif python run_predict.py \
--mode predict_dir --prefix Ind002_M_Jaw \
--pretrained_model weights/nisqa_tts.tar \
--data_dir ../HifiDiff/checkpoints/sample_slow/diffwave_step50000/ \
--num_workers 0 --bs 10 \
--output_dir results \
--filename diffwave_step50000_Ind002_M_Jaw >> logs/diffwave_step50000_Ind002_M_Jaw.log 2>&1

singularity exec -i --nv ~/pytorch_22.02-py3.sif python run_predict.py \
--mode predict_dir --prefix Ind001_F_Snd \
--pretrained_model weights/nisqa_tts.tar \
--data_dir ../HifiDiff/checkpoints/sample_slow/diffwave_step50000/ \
--num_workers 0 --bs 10 \
--output_dir results \
--filename diffwave_step50000_Ind001_F_Snd >> logs/diffwave_step50000_Ind001_F_Snd.log 2>&1

singularity exec -i --nv ~/pytorch_22.02-py3.sif python run_predict.py \
--mode predict_dir --prefix afr_ \
--pretrained_model weights/nisqa_tts.tar \
--data_dir ../HifiDiff/checkpoints/sample_slow/diffwave_step50000/ \
--num_workers 0 --bs 10 \
--output_dir results \
--filename diffwave_step50000_afr >> logs/diffwave_step50000_afr.log 2>&1

singularity exec -i --nv ~/pytorch_22.02-py3.sif python run_predict.py \
--mode predict_dir --prefix arf_ \
--pretrained_model weights/nisqa_tts.tar \
--data_dir ../HifiDiff/checkpoints/sample_slow/diffwave_step50000/ \
--num_workers 0 --bs 10 \
--output_dir results \
--filename diffwave_step50000_arf >> logs/diffwave_step50000_arf.log 2>&1

singularity exec -i --nv ~/pytorch_22.02-py3.sif python run_predict.py \
--mode predict_dir --prefix sso_ \
--pretrained_model weights/nisqa_tts.tar \
--data_dir ../HifiDiff/checkpoints/sample_slow/diffwave_step50000/ \
--num_workers 0 --bs 10 \
--output_dir results \
--filename diffwave_step50000_sso >> logs/diffwave_step50000_sso.log 2>&1

singularity exec -i --nv ~/pytorch_22.02-py3.sif python run_predict.py \
--mode predict_dir --prefix tsn_ \
--pretrained_model weights/nisqa_tts.tar \
--data_dir ../HifiDiff/checkpoints/sample_slow/diffwave_step50000/ \
--num_workers 0 --bs 10 \
--output_dir results \
--filename diffwave_step50000_tsn >> logs/diffwave_step50000_tsn.log 2>&1

singularity exec -i --nv ~/pytorch_22.02-py3.sif python run_predict.py \
--mode predict_dir --prefix xho_ \
--pretrained_model weights/nisqa_tts.tar \
--data_dir ../HifiDiff/checkpoints/sample_slow/diffwave_step50000/ \
--num_workers 0 --bs 10 \
--output_dir results \
--filename diffwave_step50000_xho >> logs/diffwave_step50000_xho.log 2>&1

singularity exec -i --nv ~/pytorch_22.02-py3.sif python run_predict.py \
--mode predict_dir --prefix xho_ \
--pretrained_model weights/nisqa_tts.tar \
--data_dir ../HifiDiff/checkpoints/sample_slow/diffwave_step50000/ \
--num_workers 0 --bs 10 \
--output_dir results \
--filename diffwave_step50000_xho >> logs/diffwave_step50000_xho.log 2>&1