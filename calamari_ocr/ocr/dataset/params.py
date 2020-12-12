from dataclasses import dataclass, field
from typing import Optional, List
from dataclasses_json import dataclass_json, config
import numpy as np

from tfaip.base.data.pipeline.definitions import Sample
from tfaip.base.data.databaseparams import DataBaseParams, DataGeneratorParams

from calamari_ocr.ocr.dataset import DataSetType
from calamari_ocr.ocr.augmentation.dataaugmentationparams import DataAugmentationAmount
from calamari_ocr.ocr.dataset.codec import Codec
from calamari_ocr.ocr.dataset.datareader.generated_line_dataset import TextGeneratorParams, LineGeneratorParams


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
    pad: int = 0
    text_index: int = 0


@dataclass
class PipelineParams(DataGeneratorParams):
    type: DataSetType = None
    skip_invalid: bool = True
    remove_invalid: bool = True
    files: List[str] = None
    text_files: Optional[List[str]] = None
    gt_extension: Optional[str] = None
    data_reader_args: Optional[FileDataReaderArgs] = None


@dataclass
class DataParams(DataBaseParams):
    train: PipelineParams = field(default_factory=PipelineParams)
    val: PipelineParams = field(default_factory=PipelineParams)
    skip_invalid_gt_: bool = True
    input_channels: int = 1
    downscale_factor_: int = -1
    line_height_: int = -1
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
            self.meta = SampleMeta()

    def to_input_target_sample(self) -> Sample:
        return Sample(self.image, self.gt, self.meta.to_dict())

