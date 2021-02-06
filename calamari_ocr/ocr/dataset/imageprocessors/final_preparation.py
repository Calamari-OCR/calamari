import numpy as np
from calamari_ocr.ocr.dataset.imageprocessors.data_preprocessor import ImageProcessor


class FinalPreparation(ImageProcessor):
    @staticmethod
    def default_params() -> dict:
        return {
            'normalize': True,
            'invert': True,
            'transpose': True,
            'pad': 0,
            'pad_value': False,
        }

    def __init__(self,
                 normalize,
                 invert,
                 transpose,
                 pad,
                 pad_value,
                 as_uint8=None,  # Deprecated
                 **kwargs):
        super().__init__(**kwargs)
        self.normalize = normalize
        self.invert = invert
        self.transpose = transpose
        self.pad = pad
        self.pad_value = pad_value

    def _apply_single(self, data, meta):
        if data.size > 0:
            # non empty image
            if self.normalize:
                amax = np.amax(data)
                if amax > 0:
                    data = data * 1.0 / amax

            if self.invert:
                data = np.amax(data) - data

        if self.transpose:
            data = np.swapaxes(data, 1, 0)

        if self.pad > 0:
            if self.transpose:
                w = data.shape[1]
                data = np.vstack([np.full((self.pad, w), self.pad_value), data, np.full((self.pad, w), self.pad_value)])
            else:
                w = data.shape[0]
                data = np.hstack([np.full((w, self.pad), self.pad_value), data, np.full((w, self.pad), self.pad_value)])

        data = (data * 255).astype(np.uint8)

        return data

    def local_to_global_pos(self, x, params):
        if self.pad > 0 and self.transpose:
            return x - self.pad
        else:
            return x
