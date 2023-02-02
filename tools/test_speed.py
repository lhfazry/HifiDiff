import sys
import importlib
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

    print(params)
    if params.model == 1:
        model = DiffWave(params)#.cuda()
    elif params.model == 2:
        model = HifiDiff(params=params)#.cuda()

    sr, audio = read('/workspace/LJSpeech-1.1/wavs/LJ001-0001.wav')
    # model inference
    model(audio, spectrogram, diffusion_step)

if __name__ == '__main__':
    parser = ArgumentParser(description='train (or resume training) a PriorGrad model')
    parser.add_argument('config1', help='config1')
    parser.add_argument('config2', help='config2')
    args = parser.parse_args()

    check_speed(args.config1)
    check_speed(args.config2)
