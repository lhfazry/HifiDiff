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


def main(args):
    results = 0
    total = 0
    sr = 22050

    for fname in glob(os.path.join(args.sdir, f"{args.prefix}*.wav")):
        print(Path(fname))
        mcd, penalty, _ = get_metrics_wavs(Path(fname), 
            Path(os.path.join(args.odir, fname)))

        results += mcd
        total += 1
    
    print(f"average: {results/total}")

    
if __name__ == '__main__':
    parser = ArgumentParser(description='Calculate MCD')
    parser.add_argument('--sdir', help='Synthetic directory of waveform')
    parser.add_argument('--odir', help='Original directory of waveform')
    parser.add_argument('--sr', type=int, default=22050, help='Sampling rate')
    parser.add_argument('--prefix', default=None, help='Prefix')
    
    main(parser.parse_args())
