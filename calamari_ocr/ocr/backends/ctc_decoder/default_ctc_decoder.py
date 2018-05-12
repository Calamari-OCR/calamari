import numpy as np

from calamari_ocr.ocr.backends.ctc_decoder.ctc_decoder import CTCDecoder
from calamari_ocr.proto import PredictionCharacter, PredictionPosition, Prediction


class DefaultCTCDecoder(CTCDecoder):
    def __init__(self, blank=0, min_p=0.0001):
        self.blank = blank
        self.threshold = min_p

        super().__init__()

    def decode(self, logits):
        last_char = self.blank
        chars = np.argmax(logits, axis=1)
        sentence = []
        for idx, c in enumerate(chars):
            if c != self.blank:
                if c != last_char:
                    sentence.append((c, idx, idx + 1))
                else:
                    _, start, end = sentence[-1]
                    del sentence[-1]
                    sentence.append((c, start, idx + 1))

            last_char = c

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
                if p[label] < self.threshold and len(pos.chars) > 0:
                    break
                else:
                    char = pos.chars.add()
                    char.label = label
                    char.probability = p[label]

        return pred

    def prob_of_sentence(self, logits):
        # do a forward pass and compute the full sentence probability
        pass


if __name__ == "__main__":
    d = DefaultCTCDecoder()
    r = d.decode(np.array(np.transpose([[0.8, 0, 0.7, 0.2, 0.1], [0.1, 0.4, 0.2, 0.7, 0.8], [0.1, 0.6, 0.1, 0.1, 0.1]])))
    print(r)