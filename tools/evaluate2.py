# need numpy==1.23.1
import os
import subprocess
import sys
import glob
import librosa
from tools.preprocess import get_mel
import numpy as np
import torch
from nnmnkwii.metrics import melcd
from argparse import ArgumentParser
from tools.preprocess import MAX_WAV_VALUE, get_mel, normalize
from params import params
from scipy.io.wavfile import read
import pyworld
from scipy.io import wavfile
import pysptk
from scipy.spatial.distance import euclidean
import os
from fastdtw import fastdtw
from mel_cepstral_distance import get_metrics_wavs, get_metrics_mels


def main(args):
    results = 0
    total = 0
    sr = 22050

    for fname in os.listdir(args.sdir):
        mcd, penalty = get_metrics_wavs(os.path.join(args.sdir, fname), os.path.join(args.odir, fname))

        results += mcd
        total += 1
    
    print(f"average: {results/total}")

    
if __name__ == '__main__':
    parser = ArgumentParser(description='Calculate MCD')
    parser.add_argument('--sdir', help='Synthetic directory of waveform')
    parser.add_argument('--odir', help='Original directory of waveform')
    
    main(parser.parse_args())
