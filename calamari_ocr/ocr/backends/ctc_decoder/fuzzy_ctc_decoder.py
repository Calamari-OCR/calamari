from calamari_ocr.ocr.backends.ctc_decoder.ctc_decoder import CTCDecoder

import numpy as np


class FuzzyCTCDecoder(CTCDecoder):
    def __init__(self, blank=0, blank_threshold=0.7, alternatives_threshold=0.0001):
        super().__init__()
        self._blank = blank
        self._blank_threshold = blank_threshold
        self._alternatives_threshold = alternatives_threshold

    def decode(self, logits):
        blanks = logits[:,self._blank] >= self._blank_threshold
        chars = np.argmax(logits, axis=1)
        sentence = []
        last_char = self._blank
        for idx in range(len(blanks)):
            c = chars[idx]
            if not blanks[idx]:
                if c != last_char or len(sentence) == 0:
                    sentence.append((c, idx, idx + 1))
                else:
                    _, start, end = sentence[-1]
                    del sentence[-1]
                    sentence.append((c, start, idx + 1))

            last_char = c

        return self.find_alternatives(logits, sentence, self._alternatives_threshold)


if __name__ == "__main__":
    d = FuzzyCTCDecoder()
    r = d.decode(np.array(np.transpose([[0.8, 0, 0.7, 0.2, 0.1], [0.1, 0.4, 0.2, 0.7, 0.8], [0.1, 0.6, 0.1, 0.1, 0.1]])))
    print(r)
