import re

from calamari_ocr.ocr.text_processing import TextProcessor, TextProcessorParams


def default_groups():
    return {
        "quotes": False,
        "spaces": False,
        "roman_digits": False,
        "ligatures": False,
        "various": False,
    }


def parse_groups(string_list):
    groups = default_groups()

    for s in map(str.lower, string_list):
        if s == "none":
            groups["quotes"] = False
            groups["spaces"] = False
            groups["roman_digits"] = False
            groups["ligatures"] = False
            groups["various"] = False
        elif s == "simple":
            groups["quotes"] = False
            groups["spaces"] = True
            groups["roman_digits"] = False
            groups["ligatures"] = False
            groups["various"] = True
        elif s == "extended":
            groups["quotes"] = True
            groups["spaces"] = True
            groups["roman_digits"] = True
            groups["ligatures"] = False
            groups["various"] = True
        elif s == "all":
            groups["quotes"] = True
            groups["spaces"] = True
            groups["roman_digits"] = True
            groups["ligatures"] = True
            groups["various"] = True
        elif s in groups:
            groups[s] = True
        else:
            raise KeyError("Unknown key '{}', allowed: {}".format(s, groups.keys()))

    return groups


def default_text_regularizer_params(params=TextProcessorParams(), groups=["simple"]):
    params.type = TextProcessorParams.TEXT_REGULARIZER

    groups = parse_groups(groups)

    def replacement(old, new, regex=False):
        r = params.replacements.add()
        r.old = old
        r.new = new
        r.regex = regex

    if groups["various"]:
        replacement("µ", "μ")    # replace micro unit with greek character
        replacement("–——", "-")  # variant length hyphens
        replacement("–—", "-")   # variant length hyphens

    if groups["quotes"]:
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

    if groups["ligatures"]:
        # compare https://en.wikipedia.org/wiki/Typographic_ligature#Ligatures_in_Unicode_(Latin_alphabets)
        replacement("Ꜳ", "AA")
        replacement("ꜳ", "aa")
        replacement("Æ", "AE")
        replacement("æ", "ae")
        replacement("Ꜵ", "AO")
        replacement("ꜵ", "ao")
        replacement("Ꜷ", "AU")
        replacement("ꜷ", "au")
        replacement("Ꜹ", "AV")
        replacement("ꜹ", "av")
        replacement("Ꜻ", "AV")
        replacement("ꜻ", "av")
        replacement("Ꜽ", "AY")
        replacement("ꜽ", "ay")
        replacement("🙰", "et")
        replacement("ﬀ", "ff")
        replacement("ﬃ", "ffi")
        replacement("ﬄ", "ffl")
        replacement("ﬂ", "fl")
        replacement("ﬁ", "fi")
        replacement("Œ", "OE")
        replacement("œ", "oe")
        replacement("Ꝏ", "OO")
        replacement("ꝏ", "oo")
        replacement("ẞ", "ſs")
        replacement("ß", "ſz")
        replacement("ﬆ", "st")
        replacement("ﬅ", "ſt")
        replacement("Ꜩ", "TZ")
        replacement("ꜩ", "tz")
        replacement("ᵫ", "ue")
        replacement("Ꝡ", "VY")
        replacement("ꝡ", "vy")

    if groups["roman_digits"]:
        replacement("Ⅰ", "I")     # expand unicode roman digits
        replacement("Ⅱ", "II")   # expand unicode roman digits
        replacement("Ⅲ", "III")   # expand unicode roman digits
        replacement("Ⅳ", "IV")   # expand unicode roman digits
        replacement("Ⅴ", "V")   # expand unicode roman digits
        replacement("Ⅵ", "VI")   # expand unicode roman digits
        replacement("Ⅶ", "VII")   # expand unicode roman digits
        replacement("Ⅷ", "VIII")   # expand unicode roman digits
        replacement("Ⅸ", "IX")   # expand unicode roman digits
        replacement("Ⅹ", "X")   # expand unicode roman digits
        replacement("Ⅺ", "XI")   # expand unicode roman digits
        replacement("Ⅻ", "XII")   # expand unicode roman digits
        replacement("Ⅼ", "L")   # expand unicode roman digits
        replacement("Ⅽ", "C")   # expand unicode roman digits
        replacement("Ⅾ", "D")   # expand unicode roman digits
        replacement("Ⅿ", "M")   # expand unicode roman digits
        replacement("ⅰ", "i")     # expand unicode roman digits
        replacement("ⅱ", "ii")   # expand unicode roman digits
        replacement("ⅲ", "iii")   # expand unicode roman digits
        replacement("ⅳ", "iv")   # expand unicode roman digits
        replacement("ⅴ", "v")   # expand unicode roman digits
        replacement("ⅵ", "vi")   # expand unicode roman digits
        replacement("ⅶ", "vii")   # expand unicode roman digits
        replacement("ⅷ", "viii")   # expand unicode roman digits
        replacement("ⅸ", "ix")   # expand unicode roman digits
        replacement("ⅹ", "x")   # expand unicode roman digits
        replacement("ⅺ", "xi")   # expand unicode roman digits
        replacement("ⅻ", "xii")   # expand unicode roman digits
        replacement("ⅼ", "l")   # expand unicode roman digits
        replacement("ⅽ", "c")   # expand unicode roman digits
        replacement("ⅾ", "d")   # expand unicode roman digits
        replacement("ⅿ", "m")   # expand unicode roman digits

    if groups["spaces"]:
        replacement(r"\s+(?u)", ' ', True)   # Multiple spaces to one
        replacement(r"\n(?u)", '', True)     # Remove line breaks
        replacement(r"^\s+(?u)", '', True)   # strip left
        replacement(r"\s+$(?u)", '', True)   # strip right

    return params


class TextRegularizer(TextProcessor):
    def __init__(self, params=default_text_regularizer_params()):
        super().__init__()
        self.params = params

    def _apply_single(self, txt):
        for replacement in self.params.replacements:
            if replacement.regex:
                txt = re.sub(replacement.old, replacement.new, txt)
            else:
                txt = txt.replace(replacement.old, replacement.new)

        return txt


if __name__ == "__main__":
    n = TextRegularizer(default_text_regularizer_params(groups=["quotes", "spaces"]))
    assert(n.apply(["“Resolve quotes”"]) == ["''Resolve quotes''"])
    assert(n.apply(["  “Resolve   spaces  ”   "]) == ["''Resolve spaces ''"])
