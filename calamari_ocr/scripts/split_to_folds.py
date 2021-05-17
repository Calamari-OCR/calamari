import argparse
import os
import shutil

import tfaip.util.logging as logging
from tfaip.data.pipeline.definitions import PipelineMode
from tqdm import tqdm

from calamari_ocr.ocr import CrossFold
from calamari_ocr.ocr.dataset.datareader.file import FileDataParams
from calamari_ocr.utils import split_all_ext, glob_all

logger = logging.logger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Write split of folds to separate directories")
    parser.add_argument(
        "--files",
        nargs="+",
        help="List all image files that shall be processed. Ground truth fils with the same "
        "base name but with '.gt.txt' as extension are required at the same location",
    )
    parser.add_argument(
        "--n_folds",
        type=int,
        required=True,
        help="The number of fold, that is the number of models to train",
    )
    parser.add_argument("--output_dir", type=str, required=True, help="Where to write the folds")
    parser.add_argument(
        "--keep_original_filename",
        action="store_true",
        help="By default the copied new files get a new 8 digit name. Use this flag to keep the "
        "original name but be aware, that this might override lines with the same name",
    )

    args = parser.parse_args()

    logger.info("Creating folds")
    images = glob_all(args.files)
    texts = [split_all_ext(p)[0] + ".gt.txt" for p in images]
    data_reader = FileDataParams(images=images, texts=texts, skip_invalid=True)
    data_reader.prepare_for_mode(PipelineMode.TRAINING)
    cross_fold = CrossFold(
        n_folds=args.n_folds,
        data_generator_params=data_reader,
        output_dir=args.output_dir,
    )

    logger.info("Copying files")
    for fold_id, fold_files in enumerate(cross_fold.folds):
        fold_out_dir = os.path.join(args.output_dir, str(fold_id))
        if not os.path.exists(fold_out_dir):
            os.makedirs(fold_out_dir)

        for file_id, file in tqdm(enumerate(fold_files), total=len(fold_files), desc=f"Fold {fold_id}"):
            img_file = file
            base, ext = split_all_ext(file)
            txt_file = base + ".gt.txt"
            output_basename = os.path.basename(base) if args.keep_original_filename else f"{fold_id:08d}"

            if os.path.exists(img_file) and os.path.exists(txt_file):
                output_file = os.path.join(fold_out_dir, f"{output_basename}{ext}")
                shutil.copyfile(img_file, output_file)

                output_file = os.path.join(fold_out_dir, f"{output_basename}.gt.txt")
                shutil.copyfile(txt_file, output_file)
            else:
                logger.info(f"Warning: Does not exist {img_file} or {txt_file}")


if __name__ == "__main__":
    main()
