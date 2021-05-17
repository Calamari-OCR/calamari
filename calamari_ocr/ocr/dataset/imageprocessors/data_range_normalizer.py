from dataclasses import dataclass
from typing import Type

import numpy as np
from paiargparse import pai_dataclass
from tfaip.data.pipeline.processor.dataprocessor import DataProcessorParams

from calamari_ocr.ocr.dataset.imageprocessors.data_preprocessor import ImageProcessor
from calamari_ocr.utils.image import to_uint8


@pai_dataclass(alt="DataRange")
@dataclass
class DataRangeProcessorParams(DataProcessorParams):
    @staticmethod
    def cls() -> Type["ImageProcessor"]:
        return DataRangeProcessor


class DataRangeProcessor(ImageProcessor[DataRangeProcessorParams]):
    def _apply_single(self, data: np.ndarray, meta):
        data = to_uint8(data)

        if data.ndim == 3:
            data = np.mean(data.astype("float32"), axis=2).astype(data.dtype)

        return data
