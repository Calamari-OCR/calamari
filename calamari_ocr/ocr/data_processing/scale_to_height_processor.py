import numpy as np
from calamari_ocr.ocr.data_processing.data_preprocessor import DataPreprocessor
from calamari_ocr.proto import DataPreprocessorParams
from scipy.ndimage import interpolation


class ScaleToHeightProcessor(DataPreprocessor):
    def __init__(self, data_preprocessor_params: DataPreprocessorParams):
        super().__init__()
        self.height = data_preprocessor_params.line_height

    def _apply_single(self, data):
        scaled = ScaleToHeightProcessor.scale_to_h(data, self.height)
        scale = scaled.shape[1] / data.shape[1]
        return scaled, (scale, )

    def local_to_global_pos(self, x, params):
        scale, = params
        return x / scale

    @staticmethod
    def scale_to_h(img, target_height, order=1, dtype=np.dtype('f'), cval=0):
        h, w = img.shape
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
