from calamari_ocr.ocr.dataset.textprocessors.text_processor import TextProcessor

from calamari_ocr.ocr.dataset.textprocessors.text_normalizer import (
    TextNormalizerProcessorParams,
)
from calamari_ocr.ocr.dataset.textprocessors.text_regularizer import (
    TextRegularizerProcessorParams,
)
from calamari_ocr.ocr.dataset.textprocessors.basic_text_processors import (
    StripTextProcessorParams,
    BidiTextProcessorParams,
)
from calamari_ocr.ocr.dataset.textprocessors.str_to_char_list import (
    StrToCharListProcessorParams,
)
from calamari_ocr.ocr.dataset.textprocessors.text_synchronizer import synchronize


def text_processor_cls(s: str):
    return globals()[s]
