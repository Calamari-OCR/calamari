import numpy as np

from calamari_ocr.ocr.dataset.imageprocessors.data_preprocessor import ImageProcessor
from calamari_ocr.utils.image import to_uint8


class DataRangeNormalizer(ImageProcessor):
    def _apply_single(self, data: np.ndarray, meta):
        data = to_uint8(data)

        if data.ndim == 3:
            data = np.mean(data.astype('float32'), axis=2).astype(data.dtype)

        return data
