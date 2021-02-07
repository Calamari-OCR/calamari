import numpy as np
import cv2 as cv
from calamari_ocr.ocr.dataset.imageprocessors.scale_to_height_processor import ScaleToHeightProcessor
from calamari_ocr.ocr.dataset.imageprocessors.data_preprocessor import ImageProcessor


class CenterNormalizer(ImageProcessor):
    @staticmethod
    def default_params() -> dict:
        return {
            'extra_params': (4, 1.0, 0.3),
        }

    def __init__(self, extra_params=(4, 1.0, 0.3), debug=False, **kwargs):
        super().__init__(**kwargs)
        self.debug = debug
        self.target_height = self.params.line_height_
        self.range, self.smoothness, self.extra = extra_params

    def _apply_single(self, data, meta):
        out, params = self.normalize(data.astype(np.uint8))
        meta['center'] = params
        return out

    def set_height(self, target_height):
        self.target_height = target_height

    def measure(self, line):
        h, w = line.shape
        smoothed = cv.GaussianBlur(line, (0, 0), sigmaX=h*self.smoothness, sigmaY=h*.5,
                                   borderType=cv.BORDER_CONSTANT)
        smoothed += .001 * cv.blur(smoothed, (w, int(h*.5)), borderType=cv.BORDER_CONSTANT)

        a = np.argmax(smoothed, axis=0).astype(np.uint16)
        kernel = cv.getGaussianKernel(int((8.*h*self.extra)+1), h*self.extra)
        center = cv.filter2D(a, cv.CV_16U, kernel, borderType=cv.BORDER_REFLECT).flatten()

        deltas = abs(np.arange(h)[:, np.newaxis] - center[np.newaxis, :])
        mad = np.mean(deltas[line != 0])
        r = int(1 + self.range * mad)

        return center, r

    def dewarp(self, img, cval=0):
        """

        Parameters
        ----------
        img image with dtype=np.uint8
        cval

        Returns image with dtype=np.uint8
        -------

        """
        if img.size == 0:
            # Empty image
            return img

        if img.ndim > 2:
            temp = (cv.cvtColor(img, cv.COLOR_BGR2GRAY) / 255).astype(np.float32)
        else:
            temp = (img / 255).astype(np.float32)
        temp = np.amax(temp) - temp
        amax = np.amax(temp)
        if amax == 0:
            # white image
            return (temp * 255).astype(np.uint8)
        inverted = temp * 1.0 / np.amax(temp)

        center, r = self.measure(inverted)

        # The actual image img is embedded into a larger image by
        # adding vertical space on top and at the bottom (padding)
        hpad = r  # this is large enough
        padded = cv.copyMakeBorder(img, hpad, hpad, 0, 0, cv.BORDER_CONSTANT, value=cval)

        center = center + hpad - r
        new_h = 2*r
        dewarped = [padded[c:c+new_h, i] for i, c in enumerate(center)]

        # transpose and convert
        dewarped = np.swapaxes(np.array(dewarped, dtype=np.uint8), 1, 0)
        return dewarped

    def normalize(self, img):
        """

        Parameters
        ----------
        img: image of type np.uint8

        Returns image of dtype np.uint8
        -------

        """
        # resize the image to a appropriate height close to the target height to speed up dewarping
        intermediate_height = int(self.target_height * 1.5)
        m1 = 1

        if intermediate_height < img.shape[0]:
            m1 = intermediate_height / img.shape[0]
            img = ScaleToHeightProcessor.scale_to_h(img, intermediate_height)

        if img.size == 0:
            cval = 1
        elif img.ndim == 2:
            cval = np.amax(img).item()
        else:
            x, y = np.unravel_index(np.argmax(np.mean(img, axis=2)), img.shape[:2])
            cval = img[x, y, :].tolist()

        dewarped = self.dewarp(img, cval=cval)

        t = dewarped.shape[0] - img.shape[0]
        # scale to target height
        scaled = ScaleToHeightProcessor.scale_to_h(dewarped, self.target_height)

        if dewarped.size == 0:
            # Empty image
            m2 = 1
        else:
            m2 = scaled.shape[1] / dewarped.shape[1]
        return scaled, (m1, m2, t)

    def local_to_global_pos(self, x, params):
        m1, m2, t = params['center']
        return x / m1 / m2
