import os
import csv
import shutil
from pathlib import Path
from argparse import ArgumentParser

def main(args):
    if args.odir is not None and not os.path.exists(args.odir):
        os.makedirs(args.odir)

    with open(args.csv) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')

        for row in csv_reader:
            print(f"Copying {row[0]}")
            shutil.copyfile(row[0], os.path.join(args.odir, Path(row[0]).name))
    
if __name__ == '__main__':
    parser = ArgumentParser(description='Resample wav')
    parser.add_argument('--csv', help='Input directory of waveform')
    parser.add_argument('--odir', help='Output directory of waveform')
    
    main(parser.parse_args())
