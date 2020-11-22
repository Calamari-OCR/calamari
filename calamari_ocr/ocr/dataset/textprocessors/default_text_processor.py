from typing import List

from tfaip.base.data.pipeline.definitions import DataProcessorFactoryParams, targets_pipeline_modes

from calamari_ocr.ocr.dataset.textprocessors import BidiTextProcessor, StripTextProcessor, TextNormalizer, \
    TextRegularizer


def default_text_pre_processors() -> List[DataProcessorFactoryParams]:
    return [
        DataProcessorFactoryParams(BidiTextProcessor.__name__, targets_pipeline_modes),
        DataProcessorFactoryParams(StripTextProcessor.__name__, targets_pipeline_modes),
        DataProcessorFactoryParams(TextNormalizer.__name__, targets_pipeline_modes),
        DataProcessorFactoryParams(TextRegularizer.__name__, targets_pipeline_modes),
    ]


def default_text_post_processors() -> List[DataProcessorFactoryParams]:
    return [
        DataProcessorFactoryParams(TextNormalizer.__name__, targets_pipeline_modes),
        DataProcessorFactoryParams(TextRegularizer.__name__, targets_pipeline_modes),
        DataProcessorFactoryParams(StripTextProcessor.__name__, targets_pipeline_modes),
        DataProcessorFactoryParams(BidiTextProcessor.__name__, targets_pipeline_modes),
    ]
