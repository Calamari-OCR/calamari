from calamari_ocr.ocr.backends.ctc_decoder.ctc_decoder import CTCDecoder

import numpy as np


class FuzzyCTCDecoder(CTCDecoder):
    def __init__(self, blank=0, blank_threshold=0.7, alternatives_threshold=0.0001):
        super().__init__()
        self._blank = blank
        self._blank_threshold = blank_threshold
        self._alternatives_threshold = alternatives_threshold

    def decode(self, probabilities):
        blanks = probabilities[:, self._blank] >= self._blank_threshold
        sentence = []
        # where blank is True 'character changes' are expected
        for idx in range(len(blanks)):
            if not blanks[idx]:
                if len(sentence) == 0:
                    sentence.append((-1, idx, idx + 1))
                else:
                    _, start, end = sentence[-1]
                    if end == idx:
                        del sentence[-1]
                        sentence.append((-1, start, idx + 1))

                    else:
                        sentence.append((-1, idx, idx + 1))

        # get the best char in each range
        sentence = [(np.argmax(np.max(probabilities[start:end], axis=0)), start, end) for _, start, end in sentence]

        return self.find_alternatives(probabilities, sentence, self._alternatives_threshold)


if __name__ == "__main__":
    d = FuzzyCTCDecoder()
    r = d.decode(np.array(np.transpose([[0.8, 0, 0.7, 0.2, 0.1], [0.1, 0.4, 0.2, 0.7, 0.8], [0.1, 0.6, 0.1, 0.1, 0.1]])))
    print(r)
