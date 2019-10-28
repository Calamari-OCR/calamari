import numpy as np

from calamari_ocr.ocr.backends.ctc_decoder.ctc_decoder import CTCDecoder


class DefaultCTCDecoder(CTCDecoder):
    def __init__(self, params, codec):
        super().__init__(params, codec)
        self.blank = params.blank_index
        self.threshold = params.min_p_threshold if params.min_p_threshold > 0 else 0.0001

    def decode(self, probabilities):
        last_char = self.blank
        chars = np.argmax(probabilities, axis=1)
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

        return self.find_alternatives(probabilities, sentence, self.threshold)

    def prob_of_sentence(self, probabilities):
        # do a forward pass and compute the full sentence probability
        pass


if __name__ == "__main__":
    d = DefaultCTCDecoder()
    r = d.decode(np.array(np.transpose([[0.8, 0, 0.7, 0.2, 0.1], [0.1, 0.4, 0.2, 0.7, 0.8], [0.1, 0.6, 0.1, 0.1, 0.1]])))
    print(r)