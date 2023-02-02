import sys
import importlib
import torch
from argparse import ArgumentParser
from models.diffwave import DiffWave
from models.hifidiff import HifiDiff
from scipy.io.wavfile import read
from preprocess import MAX_WAV_VALUE, get_mel, normalize

params = None

def check_speed(config):
    print(f"Check speed: {config}")
    print(config.replace('/', '.').replace('.py', ''))
    params = importlib.import_module(config.replace('/', '.').replace('.py', '')).params

    if params.model == 1:
        model = DiffWave(params).cuda()
    elif params.model == 2:
        model = HifiDiff(params=params).cuda()

    sr, audio = read('/workspace/LJSpeech-1.1/wavs/LJ001-0001.wav')
    audio = audio / MAX_WAV_VALUE
    audio = normalize(audio) * 0.95

    # match audio length to self.hop_size * n for evaluation
    if (audio.shape[0] % params.hop_samples) != 0:
        audio = audio[:-(audio.shape[0] % params.hop_samples)]

    audio = torch.FloatTensor(audio).cuda()
    spectrogram = get_mel(audio, params)

    audio = audio.unsqueeze(0)
    #print(audio.shape)
    #print(spectrogram.shape)

    # model inference
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    start.record()
    noise = model(audio, spectrogram, torch.tensor(1.0).cuda())
    end.record()

    print(f"result noise shape: {noise.shape}")

    torch.cuda.synchronize()
    print(f"time: {start.elapsed_time(end)}\n\n")

if __name__ == '__main__':
    parser = ArgumentParser(description='train (or resume training) a PriorGrad model')
    parser.add_argument('config1', help='config1')
    parser.add_argument('config2', help='config2')
    args = parser.parse_args()

    check_speed(args.config1)
    check_speed(args.config2)
