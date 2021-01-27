import numpy as np

from calamari_ocr.ocr.dataset.imageprocessors.data_preprocessor import ImageProcessor
import cv2 as cv


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
        scale = target_height * 1.0 / h
        target_width = np.maximum(int(scale * w), 1)
        M = np.hstack((np.eye(2, dtype=np.float32) * scale, np.zeros((2, 1), dtype=np.float32)))
        out = cv.warpAffine(img.astype(np.float32), M, (target_width, target_height),
                            flags=(cv.INTER_LINEAR), borderMode=cv.BORDER_CONSTANT, borderValue=cval)
        return out.astype(dtype)
