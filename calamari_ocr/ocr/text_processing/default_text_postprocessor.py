from calamari_ocr.ocr.text_processing import \
    MultiTextProcessor, StripTextProcessor, BidiTextProcessor, \
    TextNormalizer, TextRegularizer


class DefaultTextPostprocessor(MultiTextProcessor):
    def __init__(self):
        super().__init__(
            [
                TextNormalizer(),
                TextRegularizer(),
                StripTextProcessor(),
                BidiTextProcessor(),
            ]
        )
