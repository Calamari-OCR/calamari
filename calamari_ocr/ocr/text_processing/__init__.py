from calamari_ocr.proto import TextProcessorParams

from calamari_ocr.ocr.text_processing.text_processor import MultiTextProcessor, TextProcessor, NoopTextProcessor

from calamari_ocr.ocr.text_processing.text_normalizer import TextNormalizer, default_text_normalizer_params
from calamari_ocr.ocr.text_processing.text_regularizer import TextRegularizer, default_text_regularizer_params
from calamari_ocr.ocr.text_processing.basic_text_processors import StripTextProcessor, BidiTextProcessor
from calamari_ocr.ocr.text_processing.default_text_preprocessor import DefaultTextPreprocessor
from calamari_ocr.ocr.text_processing.default_text_postprocessor import DefaultTextPostprocessor
from calamari_ocr.ocr.text_processing.text_synchronizer import synchronize


def text_processor_from_proto(text_processor_params, pre_or_post=None):
    if text_processor_params.type == TextProcessorParams.MULTI_NORMALIZER:
        return MultiTextProcessor(
            [text_processor_from_proto(c) for c in text_processor_params.children]
        )
    elif text_processor_params.type == TextProcessorParams.DEFAULT_NORMALIZER:
        if not pre_or_post:
            raise Exception("pre or post parameter must be set to specify pre or postprocessing default")
        return {"pre": DefaultTextPreprocessor(), "post": DefaultTextPostprocessor()}[pre_or_post.lower()]
    elif text_processor_params.type == TextProcessorParams.DEFAULT_PRE_NORMALIZER:
        return DefaultTextPreprocessor()
    elif text_processor_params.type == TextProcessorParams.DEFAULT_POST_NORMALIZER:
        return DefaultTextPostprocessor()
    elif text_processor_params.type == TextProcessorParams.NOOP_NORMALIZER:
        return NoopTextProcessor()
    elif text_processor_params.type == TextProcessorParams.STRIP_NORMALIZER:
        return StripTextProcessor()
    elif text_processor_params.type == TextProcessorParams.BIDI_NORMALIZER:
        return BidiTextProcessor(text_processor_params.bidi_direction)
    elif text_processor_params.type == TextProcessorParams.TEXT_NORMALIZER:
        return TextNormalizer(text_processor_params)
    elif text_processor_params.type == TextProcessorParams.TEXT_REGULARIZER:
        return TextRegularizer(text_processor_params)

    raise Exception("Unknown proto type {} of an text processor".format(text_processor_params.type))
