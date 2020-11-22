import bidi.algorithm as bidi_algorithm

from calamari_ocr.ocr.dataset.textprocessors import TextProcessor


class StripTextProcessor(TextProcessor):
    def _apply_single(self, txt, meta):
        if isinstance(txt, str):
            return txt.strip()

        elif isinstance(txt, list):
            while txt[0].isspace():
                del txt[0]

            while txt[-1].isspace():
                del txt[-1]

            return txt

        else:
            raise TypeError()


class BidiTextProcessor(TextProcessor):
    @staticmethod
    def default_params() -> dict:
        return {'bidi_direction': None}

    def __init__(self, bidi_direction, **kwargs):
        super().__init__(**kwargs)
        self.base_dir = bidi_direction

    def _apply_single(self, txt, meta):
        # To support arabic text
        return bidi_algorithm.get_display(txt, base_dir=self.base_dir)
