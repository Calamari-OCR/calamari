from typing import List

from tfaip.data.pipeline.definitions import INPUT_PROCESSOR
from tfaip.data.pipeline.processor.dataprocessor import DataProcessorParams

from calamari_ocr.ocr.dataset.imageprocessors.center_normalizer import (
    CenterNormalizerProcessorParams,
)
from calamari_ocr.ocr.dataset.imageprocessors.data_range_normalizer import (
    DataRangeProcessorParams,
)
from calamari_ocr.ocr.dataset.imageprocessors.final_preparation import (
    FinalPreparationProcessorParams,
)


def default_image_processors() -> List[DataProcessorParams]:
    return [
        CenterNormalizerProcessorParams(modes=INPUT_PROCESSOR),
        FinalPreparationProcessorParams(modes=INPUT_PROCESSOR),
    ]
