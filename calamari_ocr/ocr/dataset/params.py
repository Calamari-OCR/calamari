import logging
from dataclasses import dataclass, field
from typing import Optional

from paiargparse import pai_dataclass, pai_meta
from tfaip.base import DataBaseParams

from calamari_ocr.ocr.dataset.codec import Codec
from calamari_ocr.ocr.dataset.datareader.abbyy.reader import Abbyy
from calamari_ocr.ocr.dataset.datareader.file import FileDataParams
from calamari_ocr.ocr.dataset.datareader.hdf5.reader import Hdf5
from calamari_ocr.ocr.dataset.datareader.pagexml.reader import PageXML



def encoder(value):
    if value is None:
        return None
    return value.to_dict()


def decoder(t):
    def _decode(value):
        if value is None:
            return None
        return t.from_dict(value)

    return _decode


DATA_GENERATOR_CHOICES = [FileDataParams, PageXML, Abbyy, Hdf5]


@pai_dataclass
@dataclass
class DataParams(DataBaseParams):
    skip_invalid_gt: bool = True
    input_channels: int = 1
    downscale_factor: int = field(default=-1, metadata=pai_meta(mode='ignore'))  # Set based on model
    line_height: int = field(default=48, metadata=pai_meta(help="The line height"))
    ensemble: int = field(default=0, metadata=pai_meta(mode='ignore'))  # Set based on model
    raw_dataset: bool = False
    codec: Optional[Codec] = field(default=None, metadata=pai_meta(mode='ignore'))

    def __post_init__(self):
        from calamari_ocr.ocr.dataset.imageprocessors.center_normalizer import CenterNormalizerParams
        from calamari_ocr.ocr.dataset.imageprocessors.scale_to_height_processor import ScaleToHeightProcessorParams
        for p in self.post_proc.processors + self.pre_proc.processors:
            if isinstance(p, ScaleToHeightProcessorParams):
                p.height = self.line_height
            elif isinstance(p, CenterNormalizerParams):
                p.line_height = self.line_height
