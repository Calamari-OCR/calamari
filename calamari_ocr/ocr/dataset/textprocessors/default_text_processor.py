from typing import List

from tfaip.data.pipeline.processor.dataprocessor import DataProcessorParams

from calamari_ocr.ocr.dataset.textprocessors import (
    TextNormalizerProcessorParams,
    TextRegularizerProcessorParams,
)
from calamari_ocr.ocr.dataset.textprocessors.basic_text_processors import (
    BidiTextProcessorParams,
    StripTextProcessorParams,
)


def default_text_pre_processors() -> List[DataProcessorParams]:
    return [
        BidiTextProcessorParams(),
        StripTextProcessorParams(),
        TextNormalizerProcessorParams(),
        TextRegularizerProcessorParams(),
    ]


def default_text_post_processors() -> List[DataProcessorParams]:
    return [
        TextNormalizerProcessorParams(),
        TextRegularizerProcessorParams(),
        StripTextProcessorParams(),
        BidiTextProcessorParams(),
    ]
