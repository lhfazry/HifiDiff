# need numpy==1.23.1
import os
import subprocess
import sys
import glob
from tools.preprocess import get_mel
import numpy as np
import torch
from argparse import ArgumentParser
from tools.preprocess import MAX_WAV_VALUE, get_mel, normalize
import os
from pathlib import Path
from mel_cepstral_distance import get_metrics_wavs, get_metrics_mels
from glob import glob
import torchcrepe
import torch


def main(args):
    mcds = 0
    pitch = 0
    periodicity = 0
    total = 0
    sr = 22050

    for fname in glob(os.path.join(args.sdir, f"{args.prefix}*.wav")):
        mcd, penalty, _ = get_metrics_wavs(Path(fname), 
            Path(os.path.join(args.odir, Path(fname).name)))

        mcds += mcd
        total += 1
        
        audio1, sr1 = torchcrepe.load.audio(fname)
        audio2, sr2 = torchcrepe.load.audio(os.path.join(args.odir, Path(fname).name))

        if audio1.shape[1] > audio2.shape[1]:
            audio1 = audio1[:,:audio2.shape[1]]
        
        if audio2.shape[1] > audio1.shape[1]:
            audio2 = audio2[:,:audio1.shape[1]]

        #print(f"Audio1: {audio1.shape}, Audio2: {audio2.shape}")

        pitch1, periodicity1 = torchcrepe.predict(audio1, sr1, 256, 50, 550,
                           'tiny', return_periodicity=True, batch_size=1024,
                           device='cuda:0')
        
        pitch2, periodicity2 = torchcrepe.predict(audio2, sr2, 256, 50, 550,
                           'tiny', return_periodicity=True, batch_size=1024,
                           device='cuda:0')

        #print(f"pitch1: {pitch1.shape}, pitch2: {pitch2.shape}")
        pitch += torch.mean(1200 * torch.log2(pitch1 / pitch2))
        periodicity += torch.sqrt(torch.nn.functional.mse_loss(periodicity1, periodicity2))
    
    print(f"MCD: {mcds/total}")
    print(f"Pitch: {pitch/total}")
    print(f"Periodicity: {periodicity/total}")

    
if __name__ == '__main__':
    parser = ArgumentParser(description='Calculate MCD')
    parser.add_argument('--sdir', help='Synthetic directory of waveform')
    parser.add_argument('--odir', help='Original directory of waveform')
    parser.add_argument('--sr', type=int, default=22050, help='Sampling rate')
    parser.add_argument('--prefix', default='LJ', help='Prefix')
    
    main(parser.parse_args())
