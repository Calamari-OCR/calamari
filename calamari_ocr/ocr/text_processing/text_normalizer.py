import unicodedata
import re

from calamari_ocr.ocr.text_processing import TextProcessor, TextProcessorParams


def default_text_normalizer_params(params=TextProcessorParams()):
    params.type = TextProcessorParams.TEXT_NORMALIZER
    params.unicode_normalization = TextProcessorParams.NFC

    def replacement(old, new, regex=False):
        r = params.replacements.add()
        r.old = old
        r.new = new
        r.regex = regex

    replacement('"', "''")   # typewriter double quote
    replacement("`", "'")    # grave accent
    replacement('“', "''")   # fancy quotes
    replacement('”', "''")   # fancy quotes
    replacement("´", "'")    # acute accent
    replacement("‘", "'")    # single quotation mark
    replacement("’", "'")    # single quotation mark
    replacement("“", "''")   # double quotation mark
    replacement("”", "''")   # double quotation mark
    replacement("“", "''")   # German quotes
    replacement("„", ",,")   # German quotes
    replacement("…", "...")  # ellipsis
    replacement("′", "'")    # prime
    replacement("″", "''")   # double prime
    replacement("‴", "'''")  # triple prime
    replacement("〃", "''")  # ditto mark
    replacement("µ", "μ")    # replace micro unit with greek character
    replacement("–——", "-")  # variant length hyphens
    replacement("–—", "-")   # variant length hyphens
    replacement("ﬂ", "fl")   # expand Unicode ligatures
    replacement("ﬁ", "fi")   # expand unicode ligatures
    replacement("ﬀ", "ff")   # expand unicode ligatures
    replacement("ﬃ", "ffi")  # expand unicode ligatures
    replacement("ﬄ", "ffl")  # expand unicode ligatures

    replacement(r"\s+(?u)", ' ', True)   # Multiple spaces to one
    replacement(r"\n(?u)", '', True)     # Remove line breaks
    replacement(r"^\s+(?u)", '', True)   # strip left
    replacement(r"\s+$(?u)", '', True)   # strip right

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
        for replacement in self.params.replacements:
            if replacement.regex:
                txt = re.sub(replacement.old, replacement.new, txt)
            else:
                txt = txt.replace(replacement.old, replacement.new)

        return txt


if __name__ == "__main__":
    n = TextNormalizer()
    assert(n.apply(["“Resolve quotes”"]) == ["''Resolve quotes''"])
    assert(n.apply(["  “Resolve   spaces  ”   "]) == ["''Resolve spaces ''"])
