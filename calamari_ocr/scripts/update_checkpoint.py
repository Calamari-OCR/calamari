import argparse
from tqdm import tqdm
import os

from calamari_ocr.ocr import Checkpoint
from calamari_ocr.utils import glob_all


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoints', nargs='+', type=str, required=True)
    parser.add_argument('--dry_run', action='store_true')

    args = parser.parse_args()

    for ckpt in tqdm(glob_all(args.checkpoints)):
        ckpt = os.path.splitext(ckpt)[0]
        Checkpoint(ckpt, dry_run=args.dry_run)


if __name__ == '__main__':
    main()
