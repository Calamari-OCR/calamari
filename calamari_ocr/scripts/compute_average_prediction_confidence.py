from argparse import ArgumentParser

from calamari_ocr.ocr.datasets import create_dataset, DataSetMode, DataSetType, InputDataset
from calamari_ocr.utils import glob_all

import numpy as np


def main():
    parser = ArgumentParser()
    parser.add_argument("--pred", nargs="+", required=True,
                        help="Extended prediction files (.json extension)")

    args = parser.parse_args()

    print("Resolving files")
    pred_files = sorted(glob_all(args.pred))

    data_set = create_dataset(
        DataSetType.EXTENDED_PREDICTION,
        DataSetMode.EVAL,
        texts=pred_files,
    )

    print('Average confidence: {:.2%}'.format(np.mean([s['best_prediction'].avg_char_probability for s in data_set.samples()])))


if __name__ == '__main__':
    main()
