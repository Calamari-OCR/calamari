import numpy as np
from calamari_ocr.ocr.data_processing.data_preprocessor import DataPreprocessor
from calamari_ocr.proto import DataPreprocessorParams


class FinalPreparation(DataPreprocessor):
    def to_dict(self) -> dict:
        d = super(FinalPreparation, self).to_dict()
        d['normalize'] = self.normalize
        d['invert'] = self.invert
        d['transpose'] = self.transpose
        d['pad'] = self.pad
        d['pad_value'] = self.pad_value
        d['as_uint8'] = self.as_uint8
        return d

    def __init__(self,
                 normalize,
                 invert,
                 transpose,
                 pad,
                 pad_value,
                 as_uint8=True):
        super().__init__()
        self.normalize = normalize
        self.invert = invert
        self.transpose = transpose
        self.pad = pad
        self.pad_value = pad_value
        self.as_uint8 = as_uint8            # To save memory!

    def _apply_single(self, data):
        if self.normalize:
            amax = np.amax(data)
            if amax > 0:
                data = data * 1.0 / amax

        if self.invert:
            data = np.amax(data) - data

        if self.transpose:
            data = data.T

        if self.pad > 0:
            if self.transpose:
                w = data.shape[1]
                data = np.vstack([np.full((self.pad, w), self.pad_value), data, np.full((self.pad, w), self.pad_value)])
            else:
                w = data.shape[0]
                data = np.hstack([np.full((w, self.pad), self.pad_value), data, np.full((w, self.pad), self.pad_value)])

        if self.as_uint8:
            data = (data * 255).astype(np.uint8)

        return data, None

    def local_to_global_pos(self, x, params):
        if self.pad > 0 and self.transpose:
            return x - self.pad
        else:
            return x
