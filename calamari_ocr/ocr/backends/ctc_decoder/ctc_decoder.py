from abc import ABC, abstractmethod

import numpy as np

from calamari_ocr.proto import PredictionCharacter, PredictionPosition, Prediction


class CTCDecoder(ABC):
    def __init__(self):
        super().__init__()

    @abstractmethod
    def decode(self, logits):
        """
        Decoding algorithm of the individual CTCDecoder. This abstract function is reimplemented
        by the DefaultCTCDecoder and the FuzzyCTCDecoder.

        Parameters
        ----------
        logits : array_like
            Prediction of the neural net to decode or shape (length x character probability).
            The blank index must be 0.

        Returns
        -------
            a Prediction object
        """
        return Prediction()

    def find_alternatives(self, logits, sentence, threshold):
        """
        Find alternatives to the decoded sentence in the logits.
        E.g. if a 'c' is decoded in the range 2 to 4, this algorithm will add all characters in the interval [2, 4] to
        the output if the confidence of the character is higher than the threshold, respectively.


        Parameters
        ----------
        logits : array_like
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
        pred.logits.rows, pred.logits.cols = logits.shape
        pred.logits.data[:] = logits.reshape([-1])
        for c, start, end in sentence:
            p = logits[start:end]
            p = np.max(p, axis=0)

            pos = pred.positions.add()
            pos.start = start
            pos.end = end

            for label in reversed(sorted(range(len(p)), key=lambda v: p[v])):
                if p[label] < threshold and len(pos.chars) > 0:
                    break
                else:
                    char = pos.chars.add()
                    char.label = label
                    char.probability = p[label]

        return pred
