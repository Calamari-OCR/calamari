from argparse import ArgumentParser
import logging

from calamari_ocr.ocr.dataset import DataSetType
from tfaip.base.data.pipeline.definitions import PipelineMode

from calamari_ocr.ocr.dataset.datareader.factory import DataReaderFactory
from calamari_ocr.utils import glob_all

import numpy as np


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    parser = ArgumentParser()
    parser.add_argument("--pred", nargs="+", required=True,
                        help="Extended prediction files (.json extension)")

    args = parser.parse_args()

    logger.info("Resolving files")
    pred_files = sorted(glob_all(args.pred))

    reader = DataReaderFactory.create_data_reader(
        DataSetType.EXTENDED_PREDICTION,
        PipelineMode.Evaluation,
        texts=pred_files
    )

    logger.info('Average confidence: {:.2%}'.format(np.mean([s['best_prediction'].avg_char_probability for s in reader.samples()])))


if __name__ == '__main__':
    main()
