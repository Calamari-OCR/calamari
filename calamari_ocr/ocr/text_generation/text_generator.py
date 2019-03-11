from calamari_ocr.proto import TextGeneratorParameters
from calamari_ocr.ocr.line_generator import Script, Word, FontVariantType
import numpy as np
from calamari_ocr.ocr.codec import Codec

class TextGenerator:
    @staticmethod
    def words_to_unformatted_text(words):
        out = ''
        for word in words:
            out += word.text

        return out

    def __init__(self, text_generator_params: TextGeneratorParameters):
        self.params = text_generator_params
        self.charset = list(text_generator_params.charset)
        self.super_charset = list(text_generator_params.super_charset) if len(text_generator_params.super_charset) > 0 else self.charset
        self.sub_charset = list(text_generator_params.sub_charset) if len(text_generator_params.sub_charset) > 0 else self.charset
        assert(self.params.sub_script_p + self.params.super_script_p <= 1)

    def generate(self):
        number_of_words = int(np.round(max(1, np.random.normal(self.params.number_of_words_mean, self.params.number_of_words_sigma))))

        out = []
        for i in range(number_of_words):
            word_length = int(np.round(max(1, np.random.normal(self.params.word_length_mean, self.params.word_length_sigma))))
            rnd = np.random.rand(10)

            if rnd[0] < self.params.sub_script_p:
                script = Script.SUB
            elif rnd[0] < self.params.sub_script_p + self.params.super_script_p:
                script = Script.SUPER
            else:
                script = Script.NORMAL

            if rnd[1] < self.params.bold_p and rnd[2] < self.params.italic_p:
                variant = FontVariantType.BOLD_ITALICS
            elif rnd[1] < self.params.bold_p:
                variant = FontVariantType.BOLD
            elif rnd[2] < self.params.italic_p:
                variant = FontVariantType.ITALIC
            else:
                variant = FontVariantType.NORMAL

            if rnd[3] < self.params.letter_spacing_p:
                letter_spacing = np.random.normal(self.params.letter_spacing_mean, self.params.letter_spacing_sigma)
            else:
                letter_spacing = 0

            if script == Script.NORMAL and len(out) > 0:
                out.append(
                    Word(self.params.word_separator, Script.NORMAL, 0, FontVariantType.NORMAL)
                )

            charset = [self.charset, self.super_charset, self.sub_charset][script]
            s = "".join(np.random.choice(charset, word_length))
            s = s.strip()

            out.append(
                Word(s, script, letter_spacing, variant)
            )

        return out


if __name__ == "__main__":
    params = TextGeneratorParameters()
    params.word_length_mean = 11
    params.word_length_sigma = 3
    params.number_of_words_mean = 7
    params.number_of_words_mean = 4
    params.word_separator = " "
    params.sub_script_p = 0.0
    params.super_script_p = 0.2
    params.letter_spacing_p = 0.5
    params.letter_spacing_mean = 1
    params.letter_spacing_sigma = 0.1
    params.charset.extend(list("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789{}[]()_-.;:'\" "))
    params.super_charset.extend(list("ABCDEFGHIJKLMNOPQRSTUVWXYZ"))
    gen = TextGenerator(params)

    print(gen.generate())
    print(gen.generate())
    print(gen.generate())

    text = gen.generate()
    print(text)
    print(TextGenerator.words_to_unformatted_text(text))

