import numpy as np
from calamari_ocr.ocr.data_processing.data_preprocessor import DataPreprocessor


class DataRangeNormalizer(DataPreprocessor):
    def __init__(self):
        super().__init__()

    def _apply_single(self, data : np.ndarray):
        """
        Read an image and returns it as a floating point array.
        The optional page number allows images from files containing multiple
        images to be addressed.  Byte and short arrays are rescaled to
        the range 0...1 (unsigned) or -1...1 (signed).
        :rtype np.array in range 0...1 (unsigned) or -1...1 (signed)
        """
        if data.dtype == np.dtype('uint8'):
            data = data / 255.0
        if data.dtype == np.dtype('int8'):
            data = data / 127.0
        elif data.dtype == np.dtype('uint16'):
            data = data / 65536.0
        elif data.dtype == np.dtype('int16'):
            data = data / 32767.0
        elif data.dtype in [np.dtype('f'), np.dtype('float32'), np.dtype('float64')]:
            pass
        elif data.dtype == bool:
            data = data.astype(np.float32)
        else:
            raise Exception("unknown image type: {}".format(data.dtype))

        if data.ndim == 3:
            data = np.mean(data, axis=2)

        return data, None