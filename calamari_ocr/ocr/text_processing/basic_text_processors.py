import bidi.algorithm as bidi_algorithm

from calamari_ocr.ocr.text_processing import TextProcessor, TextProcessorParams


class StripTextProcessor(TextProcessor):
    def __init__(self):
        super().__init__()

    def _apply_single(self, txt):
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
    def to_dict(self) -> dict:
        d = super(BidiTextProcessor, self).to_dict()
        d['bidi_direction'] = self.base_dir
        return d

    def __init__(self, bidi_direction=None):
        super().__init__()
        self.base_dir = bidi_direction

    def _apply_single(self, txt):
        # To support arabic text
        return bidi_algorithm.get_display(txt, base_dir=self.base_dir)
