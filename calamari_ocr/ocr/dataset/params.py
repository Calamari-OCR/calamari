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


@dataclass
class PipelineParams(DataGeneratorParams):
    type: Union[DataSetType, str] = None  # str if custom dataset
    skip_invalid: bool = True
    remove_invalid: bool = True
    files: List[str] = None
    text_files: Optional[List[str]] = None
    gt_extension: Optional[str] = None
    n_folds: int = -1

    def prepare_for_mode(self, mode: PipelineMode) -> 'PipelineParams':
        from calamari_ocr.ocr.dataset.datareader.factory import DataReaderFactory
        assert (self.type is not None)
        params_out = deepcopy(self)
        # Training dataset
        logger.info("Resolving input files")
        if isinstance(self.type, str):
            try:
                self.type = DataSetType.from_string(self.type)
            except ValueError:
                # Not a valid type, must be custom
                if self.type not in DataReaderFactory.CUSTOM_READERS:
                    raise KeyError(f"DataSetType {self.type} is neither a standard DataSetType or preset as custom "
                                   f"reader ({list(DataReaderFactory.CUSTOM_READERS.keys())})")
        if not isinstance(self.type, str) and self.type not in {DataSetType.RAW, DataSetType.GENERATED_LINE}:
            input_image_files = sorted(glob_all(self.files)) if self.files else None

            if not self.text_files:
                if self.gt_extension:
                    gt_txt_files = [split_all_ext(f)[0] + self.gt_extension for f in input_image_files]
                else:
                    gt_txt_files = None
            else:
                gt_txt_files = sorted(glob_all(self.text_files))
                if mode in INPUT_PROCESSOR:
                    input_image_files, gt_txt_files = keep_files_with_same_file_name(input_image_files, gt_txt_files)
                    for img, gt in zip(input_image_files, gt_txt_files):
                        if split_all_ext(os.path.basename(img))[0] != split_all_ext(os.path.basename(gt))[0]:
                            raise Exception("Expected identical basenames of file: {} and {}".format(img, gt))
                else:
                    input_image_files = None

            if mode in {PipelineMode.Training, PipelineMode.Evaluation}:
                if len(set(gt_txt_files)) != len(gt_txt_files):
                    logger.warning("Some ground truth text files occur more than once in the data set "
                                   "(ignore this warning, if this was intended).")
                if len(set(input_image_files)) != len(input_image_files):
                    logger.warning("Some images occur more than once in the data set. "
                                   "This warning should usually not be ignored.")

            params_out.files = input_image_files
            params_out.text_files = gt_txt_files
        return params_out


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


