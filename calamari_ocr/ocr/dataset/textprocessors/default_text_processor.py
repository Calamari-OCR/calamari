from typing import List

from tfaip.base.data.pipeline.processor.dataprocessor import DataProcessorParams

from calamari_ocr.ocr.dataset.textprocessors import BidiText, StripText, TextNormalizer, TextRegularizer


def default_text_pre_processors() -> List[DataProcessorParams]:
    return [
        BidiText(),
        StripText(),
        TextNormalizer(),
        TextRegularizer(),
    ]


def default_text_post_processors() -> List[DataProcessorParams]:
    return [
        TextNormalizer(),
        TextRegularizer(),
        StripText(),
        BidiText(),
    ]
