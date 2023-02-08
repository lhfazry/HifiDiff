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


def main(args):
    results = 0
    total = 0
    _logdb_const = 10.0 / np.log(10.0) * np.sqrt(2.0)
    s = 0.0
    sr = 22050

    for fname in os.listdir(args.sdir):
        x = readmgc(os.path.join(args.sdir, fname))
        y = readmgc(os.path.join(args.odir, fname))

        distance, path = fastdtw(x, y, dist=euclidean)
        distance/= (len(x) + len(y))
        pathx = list(map(lambda l: l[0], path))
        pathy = list(map(lambda l: l[1], path))
        x, y = x[pathx], y[pathy]

        frames = x.shape[0]
        total  += frames

        z = x - y
        s += np.sqrt((z * z).sum(-1)).sum()


    MCD_value = _logdb_const * float(s) / float(total)
    print(f"average: {MCD_value}")

def readmgc(filename):
    # all parameters can adjust by yourself :)
    sr, x = wavfile.read(filename)
    assert sr == 22050
    x = x.astype(np.float64)
    frame_length = 1024
    hop_length = 256  
    # Windowing
    frames = librosa.util.frame(x, frame_length=frame_length, hop_length=hop_length).astype(np.float64).T
    frames *= pysptk.blackman(frame_length)
    assert frames.shape[1] == frame_length 
    # Order of mel-cepstrum
    order = 25
    alpha = 0.41
    stage = 5
    gamma = -1.0 / stage

    mgc = pysptk.mgcep(frames, order, alpha, gamma)
    mgc = mgc.reshape(-1, order + 1)
    print("mgc of {} is ok!".format(filename))
    return mgc

def load_mels(audio_file):
    sr, audio = read(audio_file)

    if params.sample_rate != sr:
        raise ValueError(f'Invalid sample rate {sr}.')

    audio = audio / MAX_WAV_VALUE
    audio = normalize(audio) * 0.95

    # match audio length to self.hop_size * n for evaluation
    if (audio.shape[0] % params.hop_samples) != 0:
        audio = audio[:-(audio.shape[0] % params.hop_samples)]
    
    audio = torch.FloatTensor(audio)
    spectrogram = get_mel(audio, params)

    return spectrogram
    
if __name__ == '__main__':
    parser = ArgumentParser(description='Calculate MCD')
    parser.add_argument('--sdir', help='Synthetic directory of waveform')
    parser.add_argument('--odir', help='Original directory of waveform')
    
    main(parser.parse_args())
