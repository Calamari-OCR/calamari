from abc import ABC, abstractmethod
from dataclasses import field, dataclass
from typing import List

import numpy as np
from paiargparse import pai_dataclass
from tfaip.util.enum import StrEnum

from calamari_ocr.ocr.predict.params import (
    Prediction,
    PredictionPosition,
    PredictionCharacter,
)


class CTCDecoderType(StrEnum):
    Default = "default"
    TokenPassing = "token_passing"
    WordBeamSearch = "word_beam_search"


@pai_dataclass
@dataclass
class CTCDecoderParams:
    type: CTCDecoderType = CTCDecoderType.Default
    blank_index: int = 0
    min_p_threshold: float = 0

    beam_width = 25
    non_word_chars: List[str] = field(default_factory=lambda: list("0123456789[]()_.:;!?{}-'\""))

    dictionary: List[str] = field(default_factory=list)
    word_separator: str = " "


def create_ctc_decoder(codec, params: CTCDecoderParams = None):
    params = params or CTCDecoderParams()
    if params.type == CTCDecoderType.Default:
        from .default_ctc_decoder import DefaultCTCDecoder

        return DefaultCTCDecoder(params, codec)
    elif params.type == CTCDecoderType.TokenPassing:
        from .token_passing_ctc_decoder import TokenPassingCTCDecoder

        return TokenPassingCTCDecoder(params, codec)
    elif params.type == CTCDecoderType.WordBeamSearch:
        from .ctcwordbeamsearchdecoder import WordBeamSearchCTCDecoder

        return WordBeamSearchCTCDecoder(params, codec)

    raise NotImplemented


class CTCDecoder(ABC):
    def __init__(self, params, codec):
        super().__init__()
        self.params = params
        self.codec = codec

    @abstractmethod
    def decode(self, probabilities):
        """
        Decoding algorithm of the individual CTCDecoder. This abstract function is reimplemented
        by the DefaultCTCDecoder and the FuzzyCTCDecoder.

        Parameters
        ----------
        probabilities : array_like
            Prediction probabilities of the neural net to decode or shape (length x character probability).
            The blank index must be 0.

        Returns
        -------
            a Prediction object
        """
        return Prediction()

    def _prediction_from_string(self, probabilities, sentence):
        pred = Prediction()
        pred.labels[:] = self.codec.encode(sentence)
        pred.is_voted_result = False
        pred.logits = probabilities
        for c, l in zip(sentence, pred.labels):
            pred.positions.append(PredictionPosition(chars=[PredictionCharacter(label=l, char=c, probability=1.0)]))
        return pred

    def find_alternatives(self, probabilities, sentence, threshold) -> Prediction:
        """
        Find alternatives to the decoded sentence in the logits.
        E.g. if a 'c' is decoded in the range 2 to 4, this algorithm will add all characters in the interval [2, 4] to
        the output if the confidence of the character is higher than the threshold, respectively.


        Parameters
        ----------
        probabilities : array_like
            Prediction of the neural net to decode or shape (length x character probability).
            The blank index must be 0.
        sentence : list of tuple (character index, start pos, end pos)
            The decoded sentence (depends on the CTCDecoder).
            The position refer to the character position in the logits.
        threshold : float
            Minimum confidence for alternative characters to be listed.
        Returns
        -------
            a Prediction object

        """
        # find alternatives
        pred = Prediction()
        pred.labels[:] = [c for c, _, _ in sentence]
        pred.is_voted_result = False
        pred.logits = probabilities
        pred.avg_char_probability = 0
        for c, start, end in sentence:
            p = probabilities[start:end]
            p = np.max(p, axis=0)

            pos = PredictionPosition(local_start=start, local_end=end - 1)
            pred.positions.append(pos)

            for label in reversed(sorted(range(len(p)), key=lambda v: p[v])):
                if p[label] < threshold and len(pos.chars) > 0:
                    break
                else:
                    pos.chars.append(
                        PredictionCharacter(
                            label=label,
                            probability=p[label],
                        )
                    )

            if len(pos.chars) > 0:
                pred.avg_char_probability += pos.chars[0].probability

        pred.avg_char_probability /= len(pred.positions) if len(pred.positions) > 0 else 1
        return pred
