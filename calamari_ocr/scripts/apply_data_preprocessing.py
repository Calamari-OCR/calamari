import argparse
from calamari_ocr.utils import glob_all
from tqdm import tqdm
import multiprocessing

import skimage.io as skimage_io

from calamari_ocr.proto import DataPreprocessorParams
from calamari_ocr.ocr.data_processing import MultiDataProcessor, DataRangeNormalizer, FinalPreparation, CenterNormalizer


class Handler:
    def __init__(self, data_proc, dry_run):
        self.data_proc = data_proc
        self.dry_run = dry_run

    def handle_single(self, path):
        try:
            img = skimage_io.imread(path, flatten=True)
            img = self.data_proc.apply(img)

            if not self.dry_run:
                skimage_io.imsave(path, img)
        except ValueError as e:
            print(e)
            print(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--files", type=str, nargs="+", required=True,
                        help="Text files to apply text processing")
    parser.add_argument("--line_height", type=int, default=48,
                        help="The line height")
    parser.add_argument("--pad", type=int, default=16,
                        help="Padding (left right) of the line")
    parser.add_argument("--pad_value", type=int, default=1,
                        help="Padding (left right) of the line")
    parser.add_argument("--processes", type=int, default=1)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--invert", action="store_true")
    parser.add_argument("--transpose", action="store_true")
    parser.add_argument("--dry_run", action="store_true",
                        help="No not overwrite files, just run")

    args = parser.parse_args()

    params = DataPreprocessorParams()
    params.line_height = args.line_height
    params.pad = args.pad
    params.pad_value = args.pad_value
    params.no_invert = not args.invert
    params.no_transpos = not args.transpose

    data_proc = MultiDataProcessor([
        DataRangeNormalizer(),
        CenterNormalizer(params),
        FinalPreparation(params, as_uint8=True),
    ])

    print("Resolving files")
    img_files = sorted(glob_all(args.files))

    handler = Handler(data_proc, args.dry_run)

    with multiprocessing.Pool(processes=args.processes, maxtasksperchild=100) as pool:
        list(tqdm(pool.imap(handler.handle_single, img_files), desc="Processing", total=len(img_files)))


if __name__ == "__main__":
    main()
