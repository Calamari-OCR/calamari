from dataclasses import dataclass, field
from typing import Any, NamedTuple, Optional, Tuple

from calamari_ocr.ocr.augmentation.dataaugmentationparams import DataAugmentationAmount

from calamari_ocr.ocr.augmentation import DataAugmenter
from dataclasses_json import dataclass_json, config
import numpy as np
from tfaip.base.data.data_base_params import DataBaseParams

from calamari_ocr.ocr.backends.dataset.datareader.factory import DataReaderFactory
from calamari_ocr.ocr.data_processing import DataPreprocessor
from calamari_ocr.ocr.text_processing import TextProcessor

from calamari_ocr.ocr.codec import Codec


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
class CalamariDataParams(DataBaseParams):
    skip_invalid_gt_: bool = True
    input_channels: int = 1
    downscale_factor_: int = -1
    line_height_: int = -1
    raw_dataset: bool = False
    codec: Optional[Codec] = field(default=None, metadata=config(
        encoder=encoder,
        decoder=decoder(Codec),
    ))
    text_processor: Optional[TextProcessor] = field(default=None, metadata=config(
        encoder=encoder,
        decoder=decoder(TextProcessor),
    ))
    data_processor: Optional[DataPreprocessor] = field(default=None, metadata=config(
        encoder=encoder,
        decoder=decoder(DataPreprocessor),
    ))
    data_augmenter: Optional[DataAugmenter] = field(default=None, metadata=config(
        encoder=encoder,
        decoder=decoder(DataAugmenter),
    ))
    data_aug_params: Optional[DataAugmentationAmount] = field(default=None, metadata=config(
        encoder=encoder,
        decoder=decoder(DataAugmentationAmount),
    ))

    train_reader: Optional[DataReaderFactory] = field(default=None, metadata=config(
        encoder=encoder,
        decoder=decoder(DataReaderFactory),
    ))
    val_reader: Optional[DataReaderFactory] = field(default=None, metadata=config(
        encoder=encoder,
        decoder=decoder(DataReaderFactory),
    ))
    predict_reader: Optional[DataReaderFactory] = field(default=None, metadata=config(
        encoder=encoder,
        decoder=decoder(DataReaderFactory),
    ))


@dataclass_json
@dataclass
class SampleMeta:
    id: str
    preproc_info: Optional[Any] = None
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

    def to_tuple(self) -> Tuple[Optional[np.ndarray], Optional[str], Optional[SampleMeta]]:
        return self.image, self.gt, self.meta


class PreparedSample(NamedTuple):
    image: np.ndarray  # dtype float
    gt: np.ndarray
    image_len: np.int32
    gt_len: np.int32
    serialized_meta: str


