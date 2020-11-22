from calamari_ocr.ocr.dataset.textprocessors.text_processor import TextProcessor, NoopTextProcessor

from calamari_ocr.ocr.dataset.textprocessors.text_normalizer import TextNormalizer
from calamari_ocr.ocr.dataset.textprocessors.text_regularizer import TextRegularizer
from calamari_ocr.ocr.dataset.textprocessors.basic_text_processors import StripTextProcessor, BidiTextProcessor
from calamari_ocr.ocr.dataset.textprocessors.str_to_char_list import StrToCharList
from calamari_ocr.ocr.dataset.textprocessors.text_synchronizer import synchronize


def text_processor_cls(s: str):
    return globals()[s]
