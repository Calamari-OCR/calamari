import os
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Optional, List, Union
from dataclasses_json import dataclass_json, config
import numpy as np
import logging

from tfaip.base.data.pipeline.definitions import Sample, PipelineMode, INPUT_PROCESSOR
from tfaip.base.data.databaseparams import DataBaseParams, DataGeneratorParams

from calamari_ocr.ocr.dataset import DataSetType
from calamari_ocr.ocr.augmentation.dataaugmentationparams import DataAugmentationAmount
from calamari_ocr.ocr.dataset.codec import Codec
from calamari_ocr.ocr.dataset.datareader.generated_line_dataset import TextGeneratorParams, LineGeneratorParams
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


@dataclass_json
@dataclass
class FileDataReaderArgs:
    line_generator_params: Optional[LineGeneratorParams] = None
    text_generator_params: Optional[TextGeneratorParams] = None
    pad: Optional[List[int]] = 0
    text_index: int = 0


@dataclass
class PipelineParams(DataGeneratorParams):
    type: Union[DataSetType, str] = None  # str if custom dataset
    skip_invalid: bool = True
    remove_invalid: bool = True
    files: List[str] = None
    text_files: Optional[List[str]] = None
    gt_extension: Optional[str] = None
    data_reader_args: Optional[FileDataReaderArgs] = None
    n_folds: int = -1

    def prepare_for_mode(self, mode: PipelineMode) -> 'PipelineParams':
        from calamari_ocr.ocr.dataset.datareader.factory import DataReaderFactory
        assert(self.type is not None)
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


@dataclass
class DataParams(DataBaseParams):
    train: PipelineParams = field(default_factory=PipelineParams)
    val: PipelineParams = field(default_factory=PipelineParams)
    skip_invalid_gt_: bool = True
    input_channels: int = 1
    downscale_factor_: int = -1
    line_height_: int = -1
    ensemble_: int = 0
    raw_dataset: bool = False
    codec: Optional[Codec] = field(default=None, metadata=config(
        encoder=encoder,
        decoder=decoder(Codec),
    ))
    data_aug_params: Optional[DataAugmentationAmount] = field(default=None, metadata=config(
        encoder=encoder,
        decoder=decoder(DataAugmentationAmount),
    ))


@dataclass_json
@dataclass
class SampleMeta:
    id: str
    augmented: bool = False
    fold_id: int = -1


@dataclass
class InputSample:
    image: Optional[np.ndarray]  # dtype uint8
    gt: Optional[str]
    meta: Optional[SampleMeta]

    def __post_init__(self):
        if self.image is not None:
            assert(self.image.dtype == np.uint8)

        if self.gt:
            assert(type(self.gt) == str)

        if self.meta:
            assert(type(self.meta) == SampleMeta)
        else:
            self.meta = SampleMeta(None)

    def to_input_target_sample(self) -> Sample:
        return Sample(inputs=self.image, targets=self.gt, meta=self.meta.to_dict())

