import unicodedata

from calamari_ocr.ocr.text_processing import TextProcessor, TextProcessorParams


def default_text_normalizer_params(params=TextProcessorParams(), default="NFC"):
    params.type = TextProcessorParams.TEXT_NORMALIZER
    params.unicode_normalization = TextProcessorParams.UnicodeNormalizationType.Value(default.upper())

    return params


class TextNormalizer(TextProcessor):
    def __init__(self, params=default_text_normalizer_params()):
        super().__init__()
        self.params = params

    def _apply_single(self, txt):
        txt = unicodedata.normalize(
            TextProcessorParams.UnicodeNormalizationType.Name(self.params.unicode_normalization),
            txt
        )

        return txt


if __name__ == "__main__":
    n = TextNormalizer(default_text_normalizer_params(default="NFC"))
