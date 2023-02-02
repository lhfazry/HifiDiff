# need numpy==1.23.1
import os
import subprocess
import sys
import glob
import librosa
from preprocess import get_mel
import numpy as np
import torch
from nnmnkwii.metrics import melcd
from argparse import ArgumentParser
from preprocess import MAX_WAV_VALUE, get_mel, normalize
from params import params
from scipy.io.wavfile import read
import pyworld

def main(args):
    results = 0
    total = 0

    for fname in os.listdir(args.sdir):
        synthetic_mels = load_mels(os.path.join(args.sdir, fname))
        original_mels = load_mels(os.path.join(args.odir, fname))

        result = melcd(synthetic_mels.numpy(), original_mels.numpy() , lengths=None)
        print(f"{fname} ==> {result}")

        #results.append(dict(fname=fname, score=result))
        results += result
        total += 1

    print(f"average: {results/total}")

def main2(args):
    results = 0
    total = 0
    sr = 22050

    for fname in os.listdir(args.sdir):
        swav, _ = librosa.load(os.path.join(args.sdir, fname), sr=sr, mono=True)
        owav, _ = librosa.load(os.path.join(args.odir, fname), sr=sr, mono=True)
        owav = owav[:swav.shape[0]]

        swav = swav.astype(np.float64)
        owav = owav.astype(np.float64)

        f0_1, timeaxis_1 = pyworld.harvest(swav, sr, frame_period=5.0, f0_floor=71.0, f0_ceil=800.0)
        sp1 = pyworld.cheaptrick(swav, f0_1, timeaxis_1, sr, fft_size=1024)  

        f0_2, timeaxis_2 = pyworld.harvest(owav, sr, frame_period=5.0, f0_floor=71.0, f0_ceil=800.0)
        sp2 = pyworld.cheaptrick(owav, f0_2, timeaxis_2, sr, fft_size=1024)  

        # mel-cepstrum
        coded_sp_1 = pyworld.code_spectral_envelope(sp1, sr, 24)
        coded_sp_2 = pyworld.code_spectral_envelope(sp2, sr, 24)

        result = melcd(coded_sp_1, coded_sp_2 , lengths=None)
        print(f"{fname}: {result}")

        results += result
        total += 1

    print(f"average: {results/total}")

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
    
    main2(parser.parse_args())
