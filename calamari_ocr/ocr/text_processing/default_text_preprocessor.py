from calamari_ocr.ocr.text_processing import \
    MultiTextProcessor, StripTextProcessor, BidiTextProcessor, \
    TextNormalizer, TextRegularizer


def DefaultTextPreprocessor():
    return MultiTextProcessor(
        [
            BidiTextProcessor(),
            StripTextProcessor(),
            TextNormalizer(),
            TextRegularizer(),
        ]
    )
