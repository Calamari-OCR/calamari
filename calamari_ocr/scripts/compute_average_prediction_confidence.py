import logging

import numpy as np
from paiargparse import PAIArgumentParser
from tfaip.data.pipeline.definitions import PipelineMode

from calamari_ocr.ocr.dataset.datareader.extended_prediction import (
    ExtendedPredictionDataParams,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def parse_args(args=None) -> ExtendedPredictionDataParams:
    parser = PAIArgumentParser()
    parser.add_root_argument("data", ExtendedPredictionDataParams)
    return parser.parse_args(args=args).data


def run(data: ExtendedPredictionDataParams):
    logger.info("Resolving files")
    logger.info(
        "Average confidence: {:.2%}".format(
            np.mean([s["best_prediction"].avg_char_probability for s in data.create(PipelineMode.EVALUATION).samples()])
        )
    )


def main():
    run(parse_args())


if __name__ == "__main__":
    main()
