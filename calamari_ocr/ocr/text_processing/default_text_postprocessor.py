from calamari_ocr.ocr.text_processing import \
    MultiTextProcessor, StripTextProcessor, BidiTextProcessor, \
    TextNormalizer, TextRegularizer


def DefaultTextPostprocessor():
    return MultiTextProcessor(
        [
            TextNormalizer(),
            TextRegularizer(),
            StripTextProcessor(),
            BidiTextProcessor(),
        ]
    )
