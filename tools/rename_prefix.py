import sys
import os
import numpy as np
from glob import glob
from argparse import ArgumentParser
from pathlib import Path

def rename_prefix(args):
    files = glob(os.path.join(args.input_dir,  f"{args.old_prefix}*.wav"))
    
    for f in files:
        pf = Path(f)
        new_fname = pf.name.replace(args.old_prefix, args.new_prefix)
        os.rename(f, os.path.join(pf.parent, new_fname))

if __name__ == '__main__':
    parser = ArgumentParser(description='Select random')
    parser.add_argument('--input_dir', help='folder')
    parser.add_argument('--old_prefix')
    parser.add_argument('--new_prefix')
    args = parser.parse_args()

    rename_prefix(args)