import argparse
import logging

import numpy as np
from calamari_ocr.ocr.dataset import DataSetType
from tfaip.base.data.pipeline.definitions import PipelineMode
from tfaip.util.multiprocessing.parallelmap import tqdm_wrapper

from calamari_ocr.ocr.dataset.datareader.factory import DataReaderFactory
from calamari_ocr.utils import glob_all, split_all_ext

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--files", nargs="+", required=True,
                        help="List of all image files with corresponding gt.txt files")
    parser.add_argument("--dataset", type=DataSetType.from_string, choices=list(DataSetType), default=DataSetType.FILE)
    parser.add_argument("--line_height", type=int, default=48,
                        help="The line height")
    parser.add_argument("--pad", type=int, default=16,
                        help="Padding (left right) of the line")

    args = parser.parse_args()

    logger.info("Resolving files")
    image_files = glob_all(args.files)
    gt_files = [split_all_ext(p)[0] + ".gt.txt" for p in image_files]

    ds = DataReaderFactory.create_data_reader(
        args.dataset,
        PipelineMode.Training,
        images=image_files, texts=gt_files, non_existing_as_empty=True)

    logger.info(f"Loading {len(image_files)} files")
    images, texts, metas = list(zip(*map(lambda s: (s.inputs, s.targets, s.meta), tqdm_wrapper(ds.generate(), progress_bar=True, total=len(ds)))))
    statistics = {
        "n_lines": len(images),
        "chars": [len(c) for c in texts],
        "widths": [img.shape[1] / img.shape[0] * args.line_height + 2 * args.pad for img in images
                   if img is not None and img.shape[0] > 0 and img.shape[1] > 0],
        "total_line_width": 0,
        "char_counts": {},
    }

    for image, text in zip(images, texts):
        for c in text:
            if c in statistics["char_counts"]:
                statistics["char_counts"][c] += 1
            else:
                statistics["char_counts"][c] = 1

    statistics["av_line_width"] = np.average(statistics["widths"])
    statistics["max_line_width"] = np.max(statistics["widths"])
    statistics["min_line_width"] = np.min(statistics["widths"])
    statistics["total_line_width"] = np.sum(statistics["widths"])

    statistics["av_chars"] = np.average(statistics["chars"])
    statistics["max_chars"] = np.max(statistics["chars"])
    statistics["min_chars"] = np.min(statistics["chars"])
    statistics["total_chars"] = np.sum(statistics["chars"])

    statistics["av_px_per_char"] = statistics["av_line_width"] / statistics["av_chars"]
    statistics["codec_size"] = len(statistics["char_counts"])

    del statistics["chars"]
    del statistics["widths"]

    logger.info(statistics)


if __name__ == "__main__":
    main()
