from .ctc_decoder import CTCDecoder
import numpy as np
from calamari_ocr.thirdparty.ctcwordbeamsearch.WordBeamSearch import (
    wordBeamSearch,
    LanguageModel,
)


class WordBeamSearchCTCDecoder(CTCDecoder):
    def __init__(self, params, codec):
        super().__init__(params, codec)
        word_chars = set(codec.charset).difference(set(params.non_word_chars))
        word_chars = [c for c in word_chars if len(c) > 0]
        self.language_model = LanguageModel(" ".join(params.dictionary), "".join(codec.charset), "".join(word_chars))

    def decode(self, probabilities):
        if self.params.blank_index == 0:
            probabilities = np.roll(probabilities, -1, axis=1)
        r = wordBeamSearch(
            probabilities,
            self.params.beam_width if self.params.beam_width > 0 else 25,
            self.language_model,
            False,
            allowWordToWordTransition=len(self.params.word_separator) == 0,
        )
        return self._prediction_from_string(probabilities, r)
