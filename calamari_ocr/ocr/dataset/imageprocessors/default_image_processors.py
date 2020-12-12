from typing import List

from tfaip.base.data.pipeline.definitions import DataProcessorFactoryParams, INPUT_PROCESSOR

from calamari_ocr.ocr.dataset.imageprocessors.center_normalizer import CenterNormalizer
from calamari_ocr.ocr.dataset.imageprocessors.data_range_normalizer import DataRangeNormalizer
from calamari_ocr.ocr.dataset.imageprocessors.final_preparation import FinalPreparation


def default_image_processors() -> List[DataProcessorFactoryParams]:
    return [
        DataProcessorFactoryParams(DataRangeNormalizer.__name__, INPUT_PROCESSOR),
        DataProcessorFactoryParams(CenterNormalizer.__name__, INPUT_PROCESSOR),
        DataProcessorFactoryParams(FinalPreparation.__name__, INPUT_PROCESSOR),
    ]
