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
    def __init__(self, default_bidi_direction=TextProcessorParams.BIDI_AUTO):
        super().__init__()
        self.base_dir = None
        self.set_base_dir_from_enum(default_bidi_direction)

    def set_base_dir_from_enum(self, d):
        self.base_dir = {
            TextProcessorParams.BIDI_LTR: "L",
            TextProcessorParams.BIDI_RTL: "R",
            TextProcessorParams.BIDI_AUTO: None,
        }[d]

    def _apply_single(self, txt):
        # To support arabic text
        return bidi_algorithm.get_display(txt, base_dir=self.base_dir)
