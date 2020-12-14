from typing import List

from tfaip.base.data.pipeline.definitions import DataProcessorFactoryParams

from calamari_ocr.ocr.dataset.textprocessors import BidiTextProcessor, StripTextProcessor, TextNormalizer, \
    TextRegularizer


def default_text_pre_processors() -> List[DataProcessorFactoryParams]:
    return [
        DataProcessorFactoryParams(BidiTextProcessor.__name__),
        DataProcessorFactoryParams(StripTextProcessor.__name__),
        DataProcessorFactoryParams(TextNormalizer.__name__),
        DataProcessorFactoryParams(TextRegularizer.__name__),
    ]


def default_text_post_processors() -> List[DataProcessorFactoryParams]:
    return [
        DataProcessorFactoryParams(TextNormalizer.__name__),
        DataProcessorFactoryParams(TextRegularizer.__name__),
        DataProcessorFactoryParams(StripTextProcessor.__name__),
        DataProcessorFactoryParams(BidiTextProcessor.__name__),
    ]
