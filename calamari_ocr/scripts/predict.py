import argparse
import codecs
import os

from calamari_ocr.utils.glob import glob_all
from calamari_ocr.ocr.dataset import FileDataSet
from calamari_ocr.ocr import Predictor



def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("files", nargs="+",
                        help="List all image files that shall be processed")
    parser.add_argument("--checkpoint", type=str,
                        help="Path to the checkpoint without file extension")
    parser.add_argument("-j", "--processes", type=int, default=1,
                        help="Number of processes to use")
    parser.add_argument("--verbose", action="store_true",
                        help="Print additional information")

    args = parser.parse_args()

    input_image_files = sorted(glob_all(args.files))

    dataset = FileDataSet(input_image_files)

    print("Found {} files in the dataset".format(len(dataset)))
    if len(dataset) == 0:
        raise Exception("Empty dataset provided. Check your files argument (got {})!".format(args.files))

    predictor = Predictor(checkpoint=args.checkpoint)
    out, samples, codec = predictor.predict(dataset, args.processes, progress_bar=True)

    for d, sample, filepath in zip(out, samples, input_image_files):
        if args.verbose:
            print("{}: '{}'".format(sample['id'], d['sentence']))

        with codecs.open(os.path.join(os.path.dirname(filepath), sample['id'] + ".pred.txt"), 'w', 'utf-8') as f:
            f.write(d['sentence'])

    print("All files written")


if __name__ == "__main__":
    main()
