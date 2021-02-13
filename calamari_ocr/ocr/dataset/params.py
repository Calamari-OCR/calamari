import logging
import os
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Optional, List, Union

import numpy as np
from dataclasses_json import dataclass_json, config
from paiargparse import pai_dataclass, pai_meta
from tfaip.base.data.databaseparams import DataBaseParams, DataGeneratorParams
from tfaip.base.data.pipeline.definitions import Sample, PipelineMode, INPUT_PROCESSOR

from calamari_ocr.ocr.dataset import DataSetType
from calamari_ocr.ocr.dataset.codec import Codec
from calamari_ocr.ocr.dataset.datareader.abbyy.reader import Abbyy
from calamari_ocr.ocr.dataset.datareader.base import CalamariDataGeneratorParams
from calamari_ocr.ocr.dataset.datareader.file import FileDataGenerator, FileDataParams
from calamari_ocr.ocr.dataset.datareader.generated_line_dataset import TextGeneratorParams, LineGeneratorParams
from calamari_ocr.ocr.dataset.datareader.hdf5.reader import Hdf5
from calamari_ocr.ocr.dataset.datareader.pagexml.reader import PageXML
from calamari_ocr.utils import glob_all, split_all_ext, keep_files_with_same_file_name

logger = logging.getLogger(__name__)


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
    train: CalamariDataGeneratorParams = field(default_factory=FileDataParams, metadata=pai_meta(choices=DATA_GENERATOR_CHOICES))
    val: CalamariDataGeneratorParams = field(default_factory=FileDataParams, metadata=pai_meta(choices=DATA_GENERATOR_CHOICES))
    skip_invalid_gt: bool = True
    input_channels: int = 1
    downscale_factor: int = field(default=-1, metadata=pai_meta(mode='ignore'))  # Set based on model
    line_height: int = field(default=48, metadata=pai_meta(help="The line height"))
    ensemble: int = field(default=0, metadata=pai_meta(mode='ignore'))  # Set based on model
    raw_dataset: bool = False
    codec: Optional[Codec] = field(default=None, metadata=config(
        encoder=encoder,
        decoder=decoder(Codec),
    ))

    def __post_init__(self):
        from calamari_ocr.ocr.dataset.imageprocessors.center_normalizer import CenterNormalizer
        from calamari_ocr.ocr.dataset.imageprocessors.scale_to_height_processor import ScaleToHeight
        for p in self.post_proc.processors + self.pre_proc.processors:
            if isinstance(p, ScaleToHeight):
                p.height = self.line_height
            elif isinstance(p, CenterNormalizer):
                p.line_height = self.line_height


