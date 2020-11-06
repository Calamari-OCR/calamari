import unicodedata

from calamari_ocr.ocr.text_processing import TextProcessor, TextProcessorParams


def default_text_normalizer_params(params=TextProcessorParams(), default="NFC"):
    params.type = TextProcessorParams.TEXT_NORMALIZER
    params.unicode_normalization = TextProcessorParams.UnicodeNormalizationType.Value(default.upper())

    return params


class TextNormalizer(TextProcessor):
    def to_dict(self) -> dict:
        d = super(TextNormalizer, self).to_dict()
        d['unicode_normalization'] = self.unicode_normalization
        return d

    def __init__(self, unicode_normalization="NFC"):
        super().__init__()
        self.unicode_normalization = unicode_normalization

    def _apply_single(self, txt):
        txt = unicodedata.normalize(
            self.unicode_normalization,
            txt
        )

        return txt


if __name__ == "__main__":
    n = TextNormalizer(default_text_normalizer_params(default="NFC"))
