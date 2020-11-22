import unicodedata

from calamari_ocr.ocr.dataset.textprocessors import TextProcessor


class TextNormalizer(TextProcessor):
    @staticmethod
    def default_params() -> dict:
        return {'unicode_normalization': "NFC"}

    def __init__(self, unicode_normalization, **kwargs):
        super().__init__(**kwargs)
        self.unicode_normalization = unicode_normalization

    def _apply_single(self, txt, meta):
        txt = unicodedata.normalize(
            self.unicode_normalization,
            txt
        )

        return txt
