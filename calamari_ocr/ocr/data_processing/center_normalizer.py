import numpy as np
from calamari_ocr.ocr.data_processing.data_preprocessor import DataPreprocessor
from scipy.ndimage import measurements, interpolation, filters


def scale_to_h(img, target_height, order=1, dtype=np.dtype('f'), cval=0):
    h, w = img.shape
    scale = target_height * 1.0 / h
    target_width = int(scale * w)
    output = interpolation.affine_transform(
        1.0 * img,
        np.eye(2) / scale,
        order=order,
        output_shape=(target_height,target_width),
        mode='constant',
        cval=cval)

    output = np.array(output, dtype=dtype)
    return output


class CenterNormalizer(DataPreprocessor):
    def __init__(self, target_height=48, params=(4, 1.0, 0.3), debug=False):
        self.debug = debug
        self.target_height = target_height
        self.range, self.smoothness, self.extra = params
        super().__init__()

    def _apply_single(self, data):
        return self.normalize(data, cval=np.amax(data))

    def set_height(self, target_height):
        self.target_height = target_height

    def measure(self, line):
        h, w = line.shape
        smoothed = filters.gaussian_filter(line, (h * 0.5, h * self.smoothness), mode='constant')
        smoothed += 0.001 * filters.uniform_filter(smoothed, (h * 0.5, w), mode='constant')
        shape = (h, w)
        a = np.argmax(smoothed, axis=0)
        a = filters.gaussian_filter(a, h * self.extra)
        center = np.array(a, 'i')
        deltas = abs(np.arange(h)[:, np.newaxis] - center[np.newaxis, :])
        mad = np.mean(deltas[line != 0])
        r = int(1 + self.range * mad)

        return center, r

    def dewarp(self, img, cval=0, dtype=np.dtype('f')):
        temp = np.amax(img) - img
        amax = np.amax(temp)
        if amax == 0:
            # white image
            return temp

        temp = temp * 1.0 / np.amax(temp)
        center, r = self.measure(temp)
        h, w = img.shape
        # The actual image img is embedded into a larger image by
        # adding vertical space on top and at the bottom (padding)
        hpadding = r # this is large enough
        padded = np.vstack([cval * np.ones((hpadding, w)), img, cval * np.ones((hpadding, w))])
        center = center + hpadding
        dewarped = [padded[center[i] - r:center[i]+r, i] for i in range(w)]
        dewarped = np.array(dewarped, dtype=dtype).T

        return dewarped

    def normalize(self, img, order=1, dtype=np.dtype('f'), cval=0):
        # resize the image to a appropriate height close to the target height to speed up dewarping
        intermediate_height = int(self.target_height * 1.5)
        if intermediate_height < img.shape[0]:
            img = scale_to_h(img, intermediate_height, order=order, dtype=dtype, cval=cval)

        # dewarp
        dewarped = self.dewarp(img, cval=cval, dtype=dtype)

        # scale to target height
        scaled = scale_to_h(dewarped, self.target_height, order=order, dtype=dtype, cval=cval)
        return scaled

