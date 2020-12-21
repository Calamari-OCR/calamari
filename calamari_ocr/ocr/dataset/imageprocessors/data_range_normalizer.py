import numpy as np

from calamari_ocr.ocr.dataset.imageprocessors.data_preprocessor import ImageProcessor


class DataRangeNormalizer(ImageProcessor):
    def _apply_single(self, data: np.ndarray, meta):
        """
        Read an image and returns it as uint8
        The optional page number allows images from files containing multiple
        images to be addressed.  All arrays are rescaled to
        the range 0...255 (unsigned)
        """
        if data.dtype == np.dtype('uint8'):
            data = data
        elif data.dtype == np.dtype('int8'):
            data = (data.astype('int16') + 128).astype('uint8')
        elif data.dtype == np.dtype('uint16'):
            data = (data / 256).astype('uint8')
        elif data.dtype == np.dtype('int16'):
            data = ((data / 128).astype('int16') + 128).astype('uint8')
        elif data.dtype in [np.dtype('f'), np.dtype('float32'), np.dtype('float64')]:
            data = (data * 255).astype('uint8')
        elif data.dtype == bool:
            data = data.astype('uint8') * 255
        else:
            raise Exception("unknown image type: {}".format(data.dtype))

        if data.ndim == 3:
            data = np.mean(data.astype('float32'), axis=2).astype(data.dtype)

        return data
