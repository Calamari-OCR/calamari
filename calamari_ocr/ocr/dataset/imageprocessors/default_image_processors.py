from typing import List

from tfaip.base.data.pipeline.definitions import INPUT_PROCESSOR
from tfaip.base.data.pipeline.processor.dataprocessor import DataProcessorParams

from calamari_ocr.ocr.dataset.imageprocessors.center_normalizer import CenterNormalizer
from calamari_ocr.ocr.dataset.imageprocessors.data_range_normalizer import DataRange
from calamari_ocr.ocr.dataset.imageprocessors.final_preparation import FinalPreparation


def default_image_processors() -> List[DataProcessorParams]:
    return [
        DataRange(modes=INPUT_PROCESSOR),
        CenterNormalizer(modes=INPUT_PROCESSOR),
        FinalPreparation(modes=INPUT_PROCESSOR),
    ]
