from abc import ABC, abstractmethod

import numpy as np

from calamari_ocr.proto import PredictionCharacter, PredictionPosition, Prediction

class CTCDecoder(ABC):
    def __init__(self):
        super().__init__()

    @abstractmethod
    def decode(self, logits):
        pass

    def find_alternatives(self, logits, sentence, threshold):
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
