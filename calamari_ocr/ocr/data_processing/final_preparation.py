import numpy as np
from calamari_ocr.ocr.data_processing.data_preprocessor import DataPreprocessor


class FinalPreparation(DataPreprocessor):
    def __init__(self, normalize=True, invert=True, transpose=True, pad=0, pad_value=0):
        super().__init__()
        self.normalize = normalize
        self.invert = invert
        self.transpose = transpose
        self.pad = pad
        self.pad_value = pad_value

    def _apply_single(self, data):
        if self.normalize:
            data = data * 1.0 / np.amax(data)

        if self.invert:
            data = np.amax(data) - data

        if self.transpose:
            data = data.T

        if self.pad > 0:
            w = data.shape[1]
            data = np.vstack([np.full((self.pad, w), self.pad_value), data, np.full((self.pad, w), self.pad_value)])

        return data
