import os
from argparse import ArgumentParser
from glob import glob
from pathlib import Path
import librosa
import soundfile as sf


def main(args):
    if args.odir is not None and not os.path.exists(args.odir):
        os.makedirs(args.odir)

    for fname in glob(os.path.join(args.idir, "*.wav")):
        wav, sr = librosa.load(fname, sr=args.sr)
        #resample_wav = resample(wav, args.sr)
        #write(os.path.join(args.odir, Path(fname).name), args.sr, wav)

        if args.override:
            sf.write(os.path.join(args.idir, Path(fname).name), wav, sr, format='wav')
        else:
            sf.write(os.path.join(args.odir, Path(fname).name), wav, sr, format='wav')
    
if __name__ == '__main__':
    parser = ArgumentParser(description='Resample wav')
    parser.add_argument('--idir', help='Input directory of waveform')
    parser.add_argument('--odir', help='Output directory of waveform')
    parser.add_argument('--override', action='store_true')
    parser.add_argument('--sr', type=int, default=22050, help='New sr')
    
    main(parser.parse_args())
