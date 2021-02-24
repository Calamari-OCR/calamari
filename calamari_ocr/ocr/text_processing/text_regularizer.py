import re

from calamari_ocr.ocr.text_processing import TextProcessor, TextProcessorParams


def default_groups():
    return {
        "quotes": False,
        "spaces": False,
        "roman_digits": False,
        "ligatures-vocal": False,
        "ligatures-consonantal": False,
        "various": False,
        "uvius": False,
        "punctuation": False,
    }


def parse_groups(string_list):
    groups = default_groups()

    for s in map(str.lower, string_list):
        if s == "none":
            groups["quotes"] = False
            groups["spaces"] = False
            groups["roman_digits"] = False
            groups["ligatures-consonantal"] = False
            groups["ligatures-vocal"] = False
            groups["various"] = False
        elif s == "simple":
            groups["quotes"] = False
            groups["spaces"] = True
            groups["roman_digits"] = False
            groups["ligatures-consonantal"] = False
            groups["ligatures-vocal"] = False
            groups["various"] = True
        elif s == "extended":
            groups["quotes"] = True
            groups["spaces"] = True
            groups["roman_digits"] = True
            groups["ligatures-consonantal"] = False
            groups["ligatures-vocal"] = False
            groups["various"] = True
        elif s == "all":
            groups["quotes"] = True
            groups["spaces"] = True
            groups["roman_digits"] = True
            groups["ligatures-consonantal"] = True
            groups["ligatures-vocal"] = True
            groups["various"] = True
        elif s == "zpd":  # work in progress
            groups["quotes"] = True
            groups["spaces"] = True
            groups["roman_digits"] = True
            groups["ligatures-consonantal"] = True
            groups["ligatures-vocal"] = False
            groups["various"] = True
            groups["punctuation"] = True
            groups["uvius"] = True
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
        replacement("µ", "μ")  # replace micro unit with greek character
        replacement("–", "-")  # variant length hyphens
        replacement("—", "-")  # variant length hyphens

    if groups["quotes"]:
        replacement('"', "''")  # typewriter double quote
        replacement("`", "'")  # grave accent
        replacement('“', "''")  # fancy quotes
        replacement('”', "''")  # fancy quotes
        replacement("´", "'")  # acute accent
        replacement("‘", "'")  # single quotation mark
        replacement("’", "'")  # single quotation mark
        replacement("“", "''")  # double quotation mark
        replacement("”", "''")  # double quotation mark
        replacement("“", "''")  # German quotes
        replacement("„", ",,")  # German quotes
        replacement("…", "...")  # ellipsis
        replacement("′", "'")  # prime
        replacement("″", "''")  # double prime
        replacement("‴", "'''")  # triple prime
        replacement("〃", "''")  # ditto mark

    if groups["ligatures-vocal"]:
        # compare https://en.wikipedia.org/wiki/Typographic_ligature#Ligatures_in_Unicode_(Latin_alphabets)
        replacement("Ꜳ", "AA")
        replacement("ꜳ", "aa")
        replacement("Æ", "AE")
        replacement("æ", "ae")
        replacement("Ꜵ", "AO")
        replacement("ꜵ", "ao")
        replacement("Ꜷ", "AU")
        replacement("ꜷ", "au")
        replacement("Œ", "OE")
        replacement("œ", "oe")
        replacement("Ꝏ", "OO")
        replacement("ꝏ", "oo")
        replacement("ᵫ", "ue")

    if groups["ligatures-consonantal"]:
        # compare https://en.wikipedia.org/wiki/Typographic_ligature#Ligatures_in_Unicode_(Latin_alphabets)
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
        replacement("ẞ", "ſs")
        replacement("ﬆ", "st")
        replacement("ﬅ", "ſt")
        replacement("Ꜩ", "TZ")
        replacement("ꜩ", "tz")
        replacement("Ꝡ", "VY")
        replacement("ꝡ", "vy")

    if groups["roman_digits"]:
        replacement("Ⅰ", "I")  # expand unicode roman digits
        replacement("Ⅱ", "II")  # expand unicode roman digits
        replacement("Ⅲ", "III")  # expand unicode roman digits
        replacement("Ⅳ", "IV")  # expand unicode roman digits
        replacement("Ⅴ", "V")  # expand unicode roman digits
        replacement("Ⅵ", "VI")  # expand unicode roman digits
        replacement("Ⅶ", "VII")  # expand unicode roman digits
        replacement("Ⅷ", "VIII")  # expand unicode roman digits
        replacement("Ⅸ", "IX")  # expand unicode roman digits
        replacement("Ⅹ", "X")  # expand unicode roman digits
        replacement("Ⅺ", "XI")  # expand unicode roman digits
        replacement("Ⅻ", "XII")  # expand unicode roman digits
        replacement("Ⅼ", "L")  # expand unicode roman digits
        replacement("Ⅽ", "C")  # expand unicode roman digits
        replacement("Ⅾ", "D")  # expand unicode roman digits
        replacement("Ⅿ", "M")  # expand unicode roman digits
        replacement("ⅰ", "i")  # expand unicode roman digits
        replacement("ⅱ", "ii")  # expand unicode roman digits
        replacement("ⅲ", "iii")  # expand unicode roman digits
        replacement("ⅳ", "iv")  # expand unicode roman digits
        replacement("ⅴ", "v")  # expand unicode roman digits
        replacement("ⅵ", "vi")  # expand unicode roman digits
        replacement("ⅶ", "vii")  # expand unicode roman digits
        replacement("ⅷ", "viii")  # expand unicode roman digits
        replacement("ⅸ", "ix")  # expand unicode roman digits
        replacement("ⅹ", "x")  # expand unicode roman digits
        replacement("ⅺ", "xi")  # expand unicode roman digits
        replacement("ⅻ", "xii")  # expand unicode roman digits
        replacement("ⅼ", "l")  # expand unicode roman digits
        replacement("ⅽ", "c")  # expand unicode roman digits
        replacement("ⅾ", "d")  # expand unicode roman digits
        replacement("ⅿ", "m")  # expand unicode roman digits

    if groups[
        "uvius"]:  # work in progress; based on Uwe Springmann's work for the GT4HistOCR corpus (https://zenodo.org/record/1344132)
        replacement("''", "\"")

        # replace transcription errors or unwanted symbols:
        replacement("z", "ʒ")  # in those trancriptions that should not have z, but ʒ (small letter ezh, U+0292)
        replacement("Z", "Ʒ")  # in those trancriptions that should not have Z, but Ʒ (capital ezh, U+01B7)
        # replacement("¶','")       # if the pilcrow sign is not in the line image
        replacement("ꝛ", "r")  # if you don't want to preserve r rotunda, U+A75B
        replacement("I", "J")  # most Fraktur fonts have only a single glyph for I and J
        replacement("⸍", "\\")  # U+2E0D -> /, regularize transcription for virgula
        # replacement("⸍','-")      # U+2E0D -> -, may also mean hyphenation at line end
        # use flattened a above instead of similar combining diaeresis, or macron
        replacement("q̈", "qᷓ")  # replace combining diaeresis (U+0308) with flattened a above (U+1DD3, qᷓ = quam)
        replacement("&c̈", "&cᷓ")  # &cᷓ = et cetera, the final a is signalled with flattened a above (U+1DD3)
        replacement("ḡ", "gᷓ")  # U+1E21 -> g + U1DD3, ang- or gna-
        # use combining r rotunda (U+1DE3, ᷣ) instead of combining ogonek above (U+1DCE, ᷎)
        # or combining hook above (U+0309, ̉); adapt to all your combinations
        replacement("v̉", "vᷣ")  # combining hook above -> comb. r rotunda, U+1DE3
        replacement("v᷎", "vᷣ")  # combining ogonek above -> comb. r rotunda, U+1DE3
        replacement("b᷎", "bᷣ")  # combining ogonek above -> comb. r rotunda, U+1DE3
        replacement("p᷎", "pᷣ")  # combining ogonek above -> comb. r rotunda, U+1DE3
        # exception: d + comb. r rotunda is hardly visible on screen with most fonts, so use eth instead for the d + something
        replacement("d̉", "ð")  # d+comb. hook > eth, U+00F0 (CTRL-d on Linux keyboard)
        replacement("ꝟ", "vᷣ")  # U+A75F -> v with comb. r rotunda, U+1DE3
        replacement("tᷣ", "t᷑")  # comb. r above -> combining ur above, U+1DD1 (in Latin passives such as dat᷑ = datur)
        replacement("ƞ", "n")  # n with long right leg (U+019E) -> n

        # replace font dependent private use area (PUA) code points with accepted Unicodes
        # see: https://en.wikipedia.org/wiki/Medieval_Unicode_Font_Initiative (MUFI)
        # see: http://www.primaresearch.org/www/assets/tools/Special%20Characters%20in%20Aletheia.pdf (IMPACT)
        replacement("", "C̣")  # PUA E066 	LATIN CAPITAL LETTER C WITH DOT BELOW -> C + U+0323
        replacement("", "Ñ")  # PUA E1DC 	LATIN CAPITAL LETTER N WITH HIGH MACRON -> N + U+0303
        replacement("", "Q̇")  # PUA E282 	LATIN CAPITAL LETTER Q WITH DOT ABOVE -> Q + U+0307
        replacement("", "aͤ")  # PUA E42C 	LATIN SMALL LETTER A WITH LATIN SMALL LETTER E ABOVE -> a + U+0364
        replacement("", "đ")  # PUA E491    	LATIN SMALL LETTER D WITH MEDIUM-HIGH OVERLINE (ACROSS ASCENDER) -> U+0111
        replacement("", "eͣ")  # PUA E4E1 	LATIN SMALL LETTER E WITH LATIN SMALL LETTER A ABOVE -> e + U+0363
        replacement("", "m̃")  # PUA E5B8 	LATIN SMALL LETTER M WITH MEDIUM-HIGH MACRON (ABOVE CHARACTER) -> m + U+0303
        replacement("",
                    "m̃")  # PUA E5D2 	LATIN SMALL LETTER M WITH MEDIUM-HIGH OVERLINE (ABOVE CHARACTER) -> m + U+0303
        replacement("", "ñ")  # PUA E5DC 	LATIN SMALL LETTER N WITH MEDIUM-HIGH MACRON (ABOVE CHARACTER) -> ñ
        replacement("", "oͤ")  # PUA E644 	LATIN SMALL LETTER O WITH LATIN SMALL LETTER E ABOVE -> o + U+0364
        replacement("", "p̃")  # PUA E665 	LATIN SMALL LETTER P WITH MACRON -> p + combining tilde
        replacement("", "q̃")  # PUA E681 	LATIN SMALL LETTER Q WITH MACRON -> q + U+0307
        replacement("",
                    "ꝗ̃")  # PUA E68B 	LATIN SMALL LETTER Q WITH STROKE THROUGH DESCENDER AND TILDE -> U+A757 + U+0303
        replacement("", "t́")  # PUA E6E2 	LATIN SMALL LETTER T WITH ACUTE -> t + U+0301
        replacement("", "uͤ")  # PUA E72B 	LATIN SMALL LETTER U WITH LATIN SMALL LETTER E ABOVE -> u + U+0364
        replacement("", "ů")  # PUA E72D 	LATIN SMALL LETTER U WITH LATIN SMALL LETTER O ABOVE -> U+016F
        replacement("", "v́")  # PUA E73A 	LATIN SMALL LETTER V WITH ACUTE -> v + U0301
        replacement("", "yͤ")  # PUA E781 	LATIN SMALL LETTER Y WITH LATIN SMALL LETTER E ABOVE -> y + U+0364
        replacement("",
                    "ß")  # PUA E8B7 	LATIN SMALL LETTER LONG S WITH FLOURISH -> ß (check; proper replacement in some German printings)
        replacement("", "ꝟ")  # PUA E8BA 	LATIN SMALL LETTER V WITH SHORT SLASH -> U+A75F
        replacement("",
                    "q;")  # PUA E8BF 	LATIN SMALL LETTER Q LIGATED WITH FINAL ET -> q; (or qʒ, or que, as you like)
        replacement("", "ſt")  # PUA EADA 	LATIN SMALL LIGATURE LONG S DESCENDING T -> ſt
        replacement("", "ſi")  # PUA EBA2 	LATIN SMALL LIGATURE LONG S I -> ſi
        replacement("", "ſl")  # PUA EBA3 	LATIN SMALL LIGATURE LONG S L -> ſl
        replacement("", "ſp")  # PUA EBA5 	LATIN SMALL LIGATURE LONG S P -> ſp
        replacement("", "ſſ")  # PUA EBA6 	LATIN SMALL LIGATURE LONG S LONG S -> ſſ
        replacement("", "ſſi")  # PUA EBA7 	LATIN SMALL LIGATURE LONG S LONG S I -> ſſi
        replacement("",
                    "ß")  # PUA EBAC 	LATIN SMALL LIGATURE LONG S INSULAR V -> ß (check for correct meaning; could also be ſ + r rotunda)
        replacement("", "j̈")  # PUA EBE3 	LATIN SMALL LETTER J WITH DIAERESIS
        replacement("", "ck")  # PUA EEC4 	LATIN SMALL LIGATURE CK
        replacement("", "ct")  # PUA EEC5 	LATIN SMALL LIGATURE CT
        replacement("", "ft")  # PUA EECB 	LATIN SMALL LIGATURE FT -> ft
        replacement("", "pp")  # PUA EED6 	LATIN SMALL LIGATURE PP -> pp
        replacement("", "ꝓp")  # PUA EED7 	LATIN SMALL LIGATURE PP WITH FLOURISH -> U+A753 + p
        replacement("", "tz")  # PUA EEDC 	LATIN SMALL LIGATURE TZ -> tz
        replacement("", "æ")  # PUA EFA1 	LATIN SMALL LIGATURE NECKLESS A E
        #        replacement("/̃")  # PUA F00A	COMBINING HIGH MACRON WITH FIXED HEIGHT (PART-WIDTH) -> U+0303
        replacement("q", "qͥ")  # PUA F02F 	COMBINING LATIN SMALL LETTER DOTLESS I -> small letter i above (U+0365)
        replacement("", "⁊")  # PUA F158 	LATIN ABBREVIATION SIGN SMALL ET WITH STROKE -> U+204A, Tironian et
        replacement("", "ð")  # PUA F159 	LATIN ABBREVIATION SIGN SMALL DE -> eth, U+00F0
        replacement("", "?")  # PUA F160 	PUNCTUS INTERROGATIVUS -> ?
        replacement("", ":")  # PUA F161 	PUNCTUS ELEVATUS -> : (oder U+2E4E, Unicode 11.0)
        replacement("", "ꝰ")  # PUA F1A5 	LATIN ABBREVIATION SIGN SPACING BASE-LINE CAPITAL US -> U+A770
        replacement("", "ꝰ")  # PUA F1A6 	LATIN ABBREVIATION SIGN SPACING BASE-LINE US -> U+A770
        replacement("", ";")  # PUA F1AC 	LATIN ABBREVIATION SIGN SEMICOLON -> ;
        replacement("t", "t᷑")  # t + PUA F1CC 	COMBINING CURLY BAR ABOVE -> t + combining ur above (U+1DD1)
        replacement("", "i")  # PUA F220 	LATIN SMALL LETTER LONG I -> i
        replacement("", "m")  # PUA F223 	LATIN SMALL LETTER M WITH RIGHT DESCENDER -> m
        replacement("", "☙")  # PUA F2AE	?? -> U+2619 (reversed rotated floral heart bullet)
        replacement("", "℔")  # PUA F2EA	DUTCH LIBRA SIGN -> U+00A3 (pound sign)
        replacement("", "ll")  # PUA F4F9 	LATIN SMALL LIGATURE LL -> ll
        replacement("", "ſk")  # PUA F4FC 	LATIN SMALL LIGATURE LONG S K
        replacement("", "ſſt")  # PUA F4FF 	LATIN SMALL LIGATURE LONG S LONG S T
        replacement("", "aͣ")  # PUA F500 	(Latin small letter a with a above) -> a + U+0363
        replacement("", "c̃")  # PUA F501 	(Latin small letter c with macron above)
        replacement("", "ch")  # PUA F502 	(Latin small letter c ligated with latin small letter h)
        replacement("", "g̊")  # PUA F504 	(Latin small letter g with ring above)
        replacement("", "g̃")  # PUA F505 	(Latin small letter g with macron above) -> g + U+0303
        replacement("", "h̊")  # PUA F506 	(Latin small letter h with ring above) -> h + U+030A
        replacement("", "p̃")  # PUA F507 	(Latin small letter p with macron above) -> p + U+0303
        replacement("", "q̊")  # PUA F508 	(Latin small letter q with ring above) -> q + U+030A
        replacement("", "q̃;")  # PUA F509 	(Latin small letter q ligated with final et with overline) -> q+ U+0303 + ;
        replacement("", "d\'")  # PUA F50A 	(Latin small letter d with apostrophe)
        replacement("", "l\'")  # PUA F50B 	(Latin small letter l with apostrophe)
        replacement("",
                    "q́;")  # PUA F50C 	(Latin small letter q with acute accent above and semicolon on the right) -> q + U+0301 + ;
        replacement("",
                    "q́;")  # PUA F50D 	(Latin small letter q ligated with final et and acute accent) -> q + U+0301 + ;
        replacement("", "q́")  # PUA F50E 	(Latin small letter q with acute accent) -> q + U+0301
        replacement("", "q̃")  # PUA F50F 	(Latin small letter q with tilde) -> q + U+0303
        replacement("", "r̃")  # PUA F510 	(Latin small letter r with macron above) -> r + U+0303
        replacement("", "s̃")  # PUA F511 	(Latin small letter s with macron above) -> s + U+0303
        replacement("", "t᷑")  # PUA F512 	(Latin small letter t with tilde) -> t + U+1DD1
        replacement("", "v̆")  # PUA F513 	(Latin small letter v with breve) -> v + U+0306
        replacement("", "w̆")  # PUA F514 	(Latin small letter w with breve) -> w + U+0306
        replacement("", "&")  # PUA F515 	(Latin small letter e ligated with latin small letter t)
        replacement("", "z̃")  # PUA F516 	(Latin small letter z with tilde) -> z + U+0303
        replacement("", "c̃")  # PUA F517 	(Latin small letter c with tilde) -> c + U+0303
        replacement("", "r̃")  # PUA F518 	(Latin small letter r with tilde) -> r + U+0303
        replacement("", "m̃")  # PUA F519 	(Latin small letter m with tilde) -> m + U+0303
        replacement("",
                    "ꝙᷓ")  # PUA F51A 	(Latin small letter q with diagonal stroke and diaeresis) -> U+A759 + U+1DD3 (flattened a above)
        replacement("", "ð")  # PUA F51B 	(Abbreviation sign "der") -> U+00F0 (eth)
        replacement("", "zᷣ")  # PUA F51D 	(Latin small letter z with hook above) -> z + U+1DE3 (combining r rotunda)
        replacement("", "ſł")  # PUA F51E 	(Latin small ligature long s l with stroke) -> ſ + ł (U+0142; ALT-GR l)
        replacement("", "pᷓ")  # PUA F51F 	(Latin small letter p with diaeresis) - > p + U+1DD3 (flattened a above)
        replacement("", "ↄ̈")  # PUA F520 	(Latin small abbreviation sign con with diaeresis) -> U+2184 + U+0308
        replacement("", "cᷓ")  # PUA F522 	(Latin small letter c with diaeresis) -> c + U+1DD3 (flattened a above)
        replacement("", "qᷓ")  # PUA F523 	(Latin small letter q with diaeresis) -> q + U+1DD3 (flattened a above)
        replacement("", "bᷣ")  # PUA F524 	(Latin small letter b with hook above) -> b + U+1DE3 (combining r rotunda)
        replacement("", "hᷣ")  # PUA F525 	(Latin small letter h with hook above) -> h + U+1DE3
        replacement("", "pᷣ")  # PUA F526 	(Latin small letter p with hook above) -> p + U+1DE3
        replacement("", "vᷣ")  # PUA F527 	(Latin small letter v with hook above) -> v + U+1DE3
        replacement("", "yᷣ")  # PUA F52A 	(Latin small letter y with latin small letter rum above)
        replacement("", "yͭ")  # PUA F52B 	(Latin small letter y with latin small letter t above) -> t + U+036D
        replacement("", "sp")  # PUA F52C 	(Latin small ligature sp)
        replacement("", "℔")  # PUA F52D 	(Old English libra) -> U+2114
        replacement("",
                    "qᷓ;")  # PUA F52F 	(Latin small letter q ligated with final et with diaeresis) -> q + U+1DD3 + ;
        replacement("", "sᷓ")  # PUA F530 	(Latin small letter s with diaeresis) -> s + U+1DD3
        replacement("", "Ca")  # PUA F531 	(Latin ligature capital C with small a)
        replacement("", "as")  # PUA F532 	(Latin small ligature as)
        replacement("", "is")  # PUA F533 	(Latin small ligature is)
        replacement("", "us")  # PUA F534 	(Latin small ligature us)
        replacement("", "Qu")  # PUA F535 	(Latin ligature capital Q with small u)
        replacement("", "ra")  # PUA F536 	(Latin small ligature ra)
        replacement("", "ta")  # PUA F537 	(Latin small ligature ta)
        replacement("", "∵")  # PUA F538 	(Upside down asterism) -> U+2235

        # replace macron with tilde (easy to reach on keyboard; signals abbreviations; tilde and macrons often indistinguishable)
        replacement("ā", "ã")
        replacement("ē", "ẽ")
        replacement("ī", "ĩ")
        replacement("ō", "õ")
        replacement("ū", "ũ")
        replacement("c̄", "c̃")
        replacement("q̄", "q̃")
        replacement("r̄", "r̃")

    if groups["punctuation"]:
        replacement(r"(\S)(\s*)([.,:;?!])(\s*)(\S)", r"\1\3 \5",
                    True)  # remove spaces before punctuation and add one after

    if groups["spaces"]:
        replacement(r"(?u)\s+", ' ', True)  # Multiple spaces to one
        replacement(r"(?u)\n", '', True)  # Remove line breaks
        replacement(r"(?u)^\s+", '', True)  # strip left
        replacement(r"(?u)\s+$", '', True)  # strip right

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
    assert (n.apply(["“Resolve quotes”"]) == ["''Resolve quotes''"])
    assert (n.apply(["  “Resolve   spaces  ”   "]) == ["''Resolve spaces ''"])
