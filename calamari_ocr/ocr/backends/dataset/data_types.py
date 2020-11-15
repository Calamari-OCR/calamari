from dataclasses import dataclass, field
from typing import Any, NamedTuple, Optional, Tuple, List

from calamari_ocr.ocr.datasets import DataSetType
from tfaip.base.data.pipeline.definitions import BasePipelineParams, InputTargetSample

from calamari_ocr.ocr.augmentation.dataaugmentationparams import DataAugmentationAmount

from dataclasses_json import dataclass_json, config
import numpy as np
from tfaip.base.data.data_base_params import DataBaseParams

from calamari_ocr.ocr.codec import Codec
from calamari_ocr.proto.params import LineGeneratorParams, TextGeneratorParams


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
class CalamariPipelineParams(BasePipelineParams):
    type: DataSetType = None
    skip_invalid: bool = True
    remove_invalid: bool = True
    files: List[str] = None
    text_files: Optional[List[str]] = None
    gt_extension: Optional[str] = None
    data_reader_args: Optional[FileDataReaderArgs] = None


@dataclass
class CalamariDataParams(DataBaseParams):
    train: CalamariPipelineParams = field(default_factory=CalamariPipelineParams)
    val: CalamariPipelineParams = field(default_factory=CalamariPipelineParams)
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

    def to_input_target_sample(self) -> InputTargetSample:
        return InputTargetSample(self.image, self.gt, self.meta.to_dict())


class PreparedSample(NamedTuple):
    image: np.ndarray  # dtype float
    gt: np.ndarray
    image_len: np.int32
    gt_len: np.int32
    serialized_meta: str


