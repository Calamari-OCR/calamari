import numpy as np

from calamari_ocr.ocr.dataset.imageprocessors.data_preprocessor import ImageProcessor
import cv2 as cv


class ScaleToHeightProcessor(ImageProcessor):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.height = self.params.line_height_

    def _apply_single(self, data, meta):
        scaled = ScaleToHeightProcessor.scale_to_h(data, self.height)
        scale = scaled.shape[1] / data.shape[1]
        meta['scale_to_height'] = (scale, )
        return scaled

    def local_to_global_pos(self, x, params):
        scale, = params['scale_to_height']
        return x / scale

    @staticmethod
    def scale_to_h(img, target_height):
        assert img.dtype == np.uint8

        h, w = img.shape[:2]
        if h == target_height:
            return img
        if h == 0 or img.size == 0:
            # empty image
            return np.zeros(shape=(target_height, w) + img.shape[2:], dtype=img.dtype)


        scale = target_height * 1.0 / h
        target_width = np.maximum(round(scale * w), 1)
        if scale <= 1:
            # Downsampling: interpolation "area"
            return cv.resize(img, (target_width, target_height), interpolation=cv.INTER_AREA)

        else:
            # Upsampling: linear interpolation
            return cv.resize(img, (target_width, target_height), interpolation=cv.INTER_LINEAR)
