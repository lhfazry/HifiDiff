import sys
import os
import numpy as np
from glob import glob
from argparse import ArgumentParser

def select_random(args):
    files = glob(os.path.join(args.input_dir,  "*.wav"))
    selecteds = np.random.choice(files, 100)
    residuals = list(set(files) - set(selecteds))
    print(f"files size: {len(files)}")
    print(f"selecteds size: {len(selecteds)}")
    print(f"residuals size: {len(residuals)}")
    #for f in residuals:
    #    os.remove(f)

if __name__ == '__main__':
    parser = ArgumentParser(description='Select random')
    parser.add_argument('--input_dir', help='folder')
    parser.add_argument('--max', type=int, default=100)
    args = parser.parse_args()

    select_random(args)