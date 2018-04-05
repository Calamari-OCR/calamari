from calamari_ocr.ocr.text_processing import MultiTextProcessor, StripTextProcessor, BidiTextProcessor


class DefaultTextPostprocessor(MultiTextProcessor):
    def __init__(self):
        super().__init__(
            [
                StripTextProcessor(),
                BidiTextProcessor(),
            ]
        )
