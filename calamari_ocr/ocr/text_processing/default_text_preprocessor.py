from calamari_ocr.ocr.text_processing import \
    MultiTextProcessor, StripTextProcessor, BidiTextProcessor, \
    TextNormalizer, TextRegularizer


class DefaultTextPreprocessor(MultiTextProcessor):
    def __init__(self):
        super().__init__(
            [
                BidiTextProcessor(),
                StripTextProcessor(),
                TextNormalizer(),
                TextRegularizer(),
            ]
        )
