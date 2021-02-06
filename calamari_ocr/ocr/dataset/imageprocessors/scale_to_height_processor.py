import numpy as np

from calamari_ocr.ocr.dataset.imageprocessors.data_preprocessor import ImageProcessor
from scipy.ndimage import interpolation


class ScaleToHeightProcessor(ImageProcessor):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.height = self.params.line_height

    def _apply_single(self, data, meta):
        scaled = ScaleToHeightProcessor.scale_to_h(data, self.height)
        scale = scaled.shape[1] / data.shape[1]
        meta['scale_to_height'] = (scale, )
        return scaled

    def local_to_global_pos(self, x, params):
        scale, = params['scale_to_height']
        return x / scale

    @staticmethod
    def scale_to_h(img, target_height, order=1, dtype=np.dtype('f'), cval=0):
        h, w = img.shape
        if h == 0 or img.size == 0:
            # empty image
            return np.zeros(shape=(target_height, w), dtype=dtype)

        scale = target_height * 1.0 / h
        target_width = np.maximum(int(scale * w), 1)
        output = interpolation.affine_transform(
            1.0 * img,
            np.eye(2) / scale,
            order=order,
            output_shape=(target_height,target_width),
            mode='constant',
            cval=cval)

        output = np.array(output, dtype=dtype)
        return output
