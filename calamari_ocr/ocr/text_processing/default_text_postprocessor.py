from calamari_ocr.ocr.text_processing import MultiTextProcessor, StripTextProcessor, BidiTextProcessor, TextNormalizer


class DefaultTextPostprocessor(MultiTextProcessor):
    def __init__(self):
        super().__init__(
            [
                TextNormalizer(),
                StripTextProcessor(),
                BidiTextProcessor(),
            ]
        )
