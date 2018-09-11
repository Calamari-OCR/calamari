import argparse
import os
import shutil
from tqdm import tqdm

from calamari_ocr.ocr import CrossFold
from calamari_ocr.utils import split_all_ext


def main():
    parser = argparse.ArgumentParser(
        description="Write split of folds to separate directories"
    )
    parser.add_argument("--files", nargs="+",
                        help="List all image files that shall be processed. Ground truth fils with the same "
                             "base name but with '.gt.txt' as extension are required at the same location")
    parser.add_argument("--n_folds", type=int, required=True,
                        help="The number of fold, that is the number of models to train")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Where to write the folds")
    parser.add_argument("--keep_original_filename", action="store_true",
                        help="By default the copied new files get a new 8 digit name. Use this flag to keep the "
                             "original name but be aware, that this might override lines with the same name")

    args = parser.parse_args()

    print("Creating folds")
    cross_fold = CrossFold(n_folds=args.n_folds, source_files=args.files, output_dir=args.output_dir)

    print("Copying files")
    for fold_id, fold_files in enumerate(cross_fold.folds):
        fold_out_dir = os.path.join(args.output_dir, str(fold_id))
        if not os.path.exists(fold_out_dir):
            os.makedirs(fold_out_dir)

        for file_id, file in tqdm(enumerate(fold_files), total=len(fold_files), desc="Fold {}".format(fold_id)):
            img_file = file
            base, ext = split_all_ext(file)
            txt_file = base + ".gt.txt"
            output_basename = os.path.basename(base) if args.keep_original_filename else "{:08d}".format(file_id)

            if os.path.exists(img_file) and os.path.exists(txt_file):
                output_file = os.path.join(fold_out_dir, "{}{}".format(output_basename, ext))
                shutil.copyfile(img_file, output_file)

                output_file = os.path.join(fold_out_dir, "{}{}".format(output_basename, ".gt.txt"))
                shutil.copyfile(txt_file, output_file)
            else:
                print("Waring: Does not exist {} or {}".format(img_file, txt_file))


if __name__ == "__main__":
    main()

