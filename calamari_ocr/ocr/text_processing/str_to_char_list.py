from typing import List

from calamari_ocr.ocr.text_processing import TextProcessor


class StrToCharList(TextProcessor):
    @staticmethod
    def default_params() -> dict:
        return {'chars': []}

    def __init__(self, chars: List[str], **kwargs):
        super().__init__(**kwargs)
        # chars are priority ordered and might be words as-well!
        self.chars = chars

    def _apply_single(self, txt, meta):
        index = 0
        out = []
        while index < len(txt):
            found = False
            for char in self.chars:
                if len(char) == 0:
                    continue  # blank
                if txt[index:index+len(char)] == char:
                    out.append(char)
                    index += len(char)
                    found = True
                    break

            if found:
                continue

            else:
                raise Exception("Could not parse remainder '{}' of '{}'".format(txt[index:], txt))

        return out

