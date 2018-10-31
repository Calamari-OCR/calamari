from calamari_ocr.ocr.text_processing import TextProcessor, TextProcessorParams


class StrToCharList(TextProcessor):
    def __init__(self, params: TextProcessorParams):
        super().__init__()
        # chars are priority ordered and might be words as-well!
        self.chars = params.characters

    def _apply_single(self, txt):
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

