from dataclasses import dataclass, field
from typing import Optional

from paiargparse import pai_dataclass, pai_meta
from tfaip import DataBaseParams

from calamari_ocr.ocr.dataset.codec import Codec
from calamari_ocr.ocr.dataset.datareader.abbyy.reader import Abbyy
from calamari_ocr.ocr.dataset.datareader.file import FileDataParams
from calamari_ocr.ocr.dataset.datareader.hdf5.reader import Hdf5
from calamari_ocr.ocr.dataset.datareader.pagexml.reader import PageXML


DATA_GENERATOR_CHOICES = [FileDataParams, PageXML, Abbyy, Hdf5]


@pai_dataclass
@dataclass
class DataParams(DataBaseParams):
    skip_invalid_gt: bool = True
    input_channels: int = 1
    downscale_factor: int = field(default=-1, metadata=pai_meta(mode="ignore"))  # Set based on model
    line_height: int = field(default=48, metadata=pai_meta(help="The line height"))
    ensemble: int = field(default=0, metadata=pai_meta(mode="ignore"))  # Set based on model
    codec: Optional[Codec] = field(default=None, metadata=pai_meta(mode="ignore"))

    @staticmethod
    def cls():
        from calamari_ocr.ocr.dataset.data import Data

        return Data

    def __post_init__(self):
        from calamari_ocr.ocr.dataset.imageprocessors.center_normalizer import (
            CenterNormalizerProcessorParams,
        )
        from calamari_ocr.ocr.dataset.imageprocessors.scale_to_height_processor import (
            ScaleToHeightProcessorParams,
        )

        for p in self.post_proc.processors + self.pre_proc.processors:
            if isinstance(p, ScaleToHeightProcessorParams):
                p.height = self.line_height
            elif isinstance(p, CenterNormalizerProcessorParams):
                p.line_height = self.line_height
