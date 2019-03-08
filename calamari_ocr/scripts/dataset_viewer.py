import matplotlib.pyplot as plt
import argparse
from calamari_ocr.ocr.datasets import create_dataset, DataSetType, DataSetMode
from calamari_ocr import __version__
from calamari_ocr.utils import glob_all, split_all_ext, keep_files_with_same_file_name
import numpy as np
import os

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--version', action='version', version='%(prog)s v' + __version__)
    parser.add_argument("--files", nargs="+",
                        help="List all image files that shall be processed. Ground truth fils with the same "
                             "base name but with '.gt.txt' as extension are required at the same location",
                        required=True)
    parser.add_argument("--text_files", nargs="+", default=None,
                        help="Optional list of GT files if they are in other directory")
    parser.add_argument("--gt_extension", default=None,
                        help="Default extension of the gt files (expected to exist in same dir)")
    parser.add_argument("--dataset", type=DataSetType.from_string, choices=list(DataSetType), default=DataSetType.FILE)

    parser.add_argument("--n_cols", type=int, default=1)
    parser.add_argument("--n_rows", type=int, default=5)
    parser.add_argument("--select", type=int, nargs="+", default=[])

    args = parser.parse_args()

    print("Resolving input files")
    input_image_files = sorted(glob_all(args.files))
    if not args.text_files:
        if args.gt_extension:
            gt_txt_files = [split_all_ext(f)[0] + args.gt_extension for f in input_image_files]
        else:
            gt_txt_files = [None] * len(input_image_files)
    else:
        gt_txt_files = sorted(glob_all(args.text_files))
        input_image_files, gt_txt_files = keep_files_with_same_file_name(input_image_files, gt_txt_files)
        for img, gt in zip(input_image_files, gt_txt_files):
            if split_all_ext(os.path.basename(img))[0] != split_all_ext(os.path.basename(gt))[0]:
                raise Exception("Expected identical basenames of file: {} and {}".format(img, gt))

    if len(set(gt_txt_files)) != len(gt_txt_files):
        raise Exception("Some image are occurring more than once in the data set.")

    dataset = create_dataset(
        args.dataset,
        DataSetMode.TRAIN,
        images=input_image_files,
        texts=gt_txt_files,
    )
    print("Found {} files in the dataset".format(len(dataset)))

    if len(args.select) == 0:
        args.select = range(len(dataset.samples()))
        samples = dataset.samples()
    else:
        samples = [dataset.samples()[i] for i in args.select]

    f, ax = plt.subplots(args.n_rows, args.n_cols, sharex='all', sharey='all')
    row, col = 0, 0
    for i, (id, sample) in enumerate(zip(args.select, samples)):
        dataset.load_single_sample(sample)
        if args.n_cols == 1:
            ax[row].imshow(sample['image'].transpose())
            ax[row].set_title("ID: {}\n{}".format(id, sample['text']))
        else:
            ax[row, col].imshow(sample['image'].transpose())
            ax[row, col].set_title("ID: {}\n{}".format(id, sample['text']))

        row += 1
        if row == args.n_rows:
            row = 0
            col += 1

        if col == args.n_cols or i == len(samples) - 1:
            plt.show()
            f, ax = plt.subplots(args.n_rows, args.n_cols, sharex='all', sharey='all')
            row, col = 0, 0


if __name__ == "__main__":
    main()
