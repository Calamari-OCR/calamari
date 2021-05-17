from dataclasses import dataclass, field
from typing import Type

import cv2 as cv
import numpy as np
from paiargparse import pai_dataclass
from tfaip.data.pipeline.processor.dataprocessor import DataProcessorParams

from calamari_ocr.ocr.dataset.imageprocessors.data_preprocessor import ImageProcessor


@pai_dataclass(alt="ScaleToHeight")
@dataclass
class ScaleToHeightProcessorParams(DataProcessorParams):
    height: int = field(default=-1)

    @staticmethod
    def cls() -> Type["ImageProcessor"]:
        return ScaleToHeightProcessor


class ScaleToHeightProcessor(ImageProcessor[ScaleToHeightProcessorParams]):
    def _apply_single(self, data, meta):
        assert self.params.height > 0  # Not initialized
        scaled = scale_to_h(data, self.params.height)
        scale = scaled.shape[1] / data.shape[1]
        meta["scale_to_height"] = (scale,)
        return scaled

    def local_to_global_pos(self, x, params):
        (scale,) = params["scale_to_height"]
        return x / scale


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
        # Down-Sampling: interpolation "area"
        return cv.resize(img, (target_width, target_height), interpolation=cv.INTER_AREA)

    else:
        # Up-Sampling: linear interpolation
        return cv.resize(img, (target_width, target_height), interpolation=cv.INTER_LINEAR)
