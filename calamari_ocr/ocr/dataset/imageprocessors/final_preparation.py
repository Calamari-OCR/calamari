from dataclasses import dataclass, field
from typing import Type

import numpy as np
from paiargparse import pai_dataclass, pai_meta
from tfaip.base.data.pipeline.processor.dataprocessor import DataProcessorParams

from calamari_ocr.ocr.dataset.imageprocessors.data_preprocessor import ImageProcessor


@pai_dataclass
@dataclass
class FinalPreparation(DataProcessorParams):
    normalize: bool = True
    invert: bool = True
    transpose: bool = True
    pad: int = field(default=16, metadata=pai_meta(
        help="Padding (left right) of the line"
    ))
    pad_value: int = 0

    @staticmethod
    def cls() -> Type['ImageProcessor']:
        return Impl


class Impl(ImageProcessor[FinalPreparation]):
    def _apply_single(self, data, meta):
        if data.size > 0:
            # non empty image
            if self.params.normalize:
                amax = np.amax(data)
                if amax > 0:
                    data = data * 1.0 / amax

            if self.params.invert:
                data = np.amax(data) - data

        if self.params.transpose:
            data = np.swapaxes(data, 1, 0)

        if self.params.pad > 0:
            if self.params.transpose:
                w = data.shape[1]
                data = np.vstack([np.full((self.params.pad, w), self.params.pad_value), data,
                                  np.full((self.params.pad, w), self.params.pad_value)])
            else:
                w = data.shape[0]
                data = np.hstack([np.full((w, self.params.pad), self.params.pad_value), data,
                                  np.full((w, self.params.pad), self.params.pad_value)])

        data = (data * 255).astype(np.uint8)

        return data

    def local_to_global_pos(self, x, params):
        if self.params.pad > 0 and self.params.transpose:
            return x - self.params.pad
        else:
            return x
