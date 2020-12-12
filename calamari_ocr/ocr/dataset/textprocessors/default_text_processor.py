from typing import List

from tfaip.base.data.pipeline.definitions import DataProcessorFactoryParams, TARGETS_PROCESSOR

from calamari_ocr.ocr.dataset.textprocessors import BidiTextProcessor, StripTextProcessor, TextNormalizer, \
    TextRegularizer


def default_text_pre_processors() -> List[DataProcessorFactoryParams]:
    return [
        DataProcessorFactoryParams(BidiTextProcessor.__name__, TARGETS_PROCESSOR),
        DataProcessorFactoryParams(StripTextProcessor.__name__, TARGETS_PROCESSOR),
        DataProcessorFactoryParams(TextNormalizer.__name__, TARGETS_PROCESSOR),
        DataProcessorFactoryParams(TextRegularizer.__name__, TARGETS_PROCESSOR),
    ]


def default_text_post_processors() -> List[DataProcessorFactoryParams]:
    return [
        DataProcessorFactoryParams(TextNormalizer.__name__, TARGETS_PROCESSOR),
        DataProcessorFactoryParams(TextRegularizer.__name__, TARGETS_PROCESSOR),
        DataProcessorFactoryParams(StripTextProcessor.__name__, TARGETS_PROCESSOR),
        DataProcessorFactoryParams(BidiTextProcessor.__name__, TARGETS_PROCESSOR),
    ]
