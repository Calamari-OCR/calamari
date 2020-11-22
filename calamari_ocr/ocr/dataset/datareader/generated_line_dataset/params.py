from dataclasses import dataclass, field
from typing import List

from dataclasses_json import dataclass_json


@dataclass_json
@dataclass
class LineGeneratorParams:
    fonts: List[str] = field(default_factory=lambda: ['Junicode.ttf', 'DejaVuSerif.ttf'])
    font_size: int = 32
    min_script_offset: float = -0.5
    max_script_offset: float = 0.5


@dataclass_json
@dataclass
class TextGeneratorParams:
    word_length_mean: float = 11
    word_length_sigma: float = 3

    charset: List[str] = field(default_factory=lambda: list("ABCEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789{}[]()_-.;:'\" "))
    super_charset: List[str] = field(default_factory=list)
    sub_charset: List[str] = field(default_factory=list)

    number_of_words_mean: float = 7
    number_of_words_sigma: float = 4
    word_separator: str = " "

    sub_script_p: float = 0
    super_script_p: float = 0
    bold_p: float = 0
    italic_p: float = 0

    letter_spacing_p: float = 0.5
    letter_spacing_mean: float = 1
    letter_spacing_sigma: float = 0.1
