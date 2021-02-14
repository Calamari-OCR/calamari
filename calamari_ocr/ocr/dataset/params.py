import logging
from dataclasses import dataclass, field
from random import shuffle
from typing import Optional

from paiargparse import pai_dataclass, pai_meta
from tfaip.base import DataGeneratorParams
from tfaip.base.data.databaseparams import DataBaseParams, TrainValGeneratorParamsBase, \
    TrainValGeneratorParams
from tfaip.base.data.pipeline.definitions import PipelineMode

from calamari_ocr.ocr.dataset.codec import Codec
from calamari_ocr.ocr.dataset.datareader.abbyy.reader import Abbyy
from calamari_ocr.ocr.dataset.datareader.base import CalamariDataGeneratorParams
from calamari_ocr.ocr.dataset.datareader.file import FileDataParams
from calamari_ocr.ocr.dataset.datareader.hdf5.reader import Hdf5
from calamari_ocr.ocr.dataset.datareader.pagexml.reader import PageXML

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
class CalamariDefaultTrainValGeneratorParams(TrainValGeneratorParams):
    train: CalamariDataGeneratorParams = field(default_factory=FileDataParams,
                                               metadata=pai_meta(choices=DATA_GENERATOR_CHOICES))
    val: CalamariDataGeneratorParams = field(default_factory=FileDataParams,
                                             metadata=pai_meta(choices=DATA_GENERATOR_CHOICES))


@pai_dataclass
@dataclass
class CalamariTrainOnlyGeneratorParams(TrainValGeneratorParamsBase):
    def train_gen(self) -> DataGeneratorParams:
        return self.train

    def val_gen(self) -> Optional[DataGeneratorParams]:
        return None

    train: CalamariDataGeneratorParams = field(default_factory=FileDataParams,
                                               metadata=pai_meta(choices=DATA_GENERATOR_CHOICES))


@pai_dataclass
@dataclass
class CalamariSplitTrainValGeneratorParams(TrainValGeneratorParams):
    train: FileDataParams = field(default_factory=FileDataParams, metadata=pai_meta(
        choices=[FileDataParams], enforce_choices=True,
    ))
    validation_split_ratio: float = field(default=0.2, metadata=pai_meta(
        help="Use factor of n of the training dataset for validation."))

    val: Optional[FileDataParams] = field(default=None, metadata=pai_meta(mode="ignore"))

    def __post_init__(self):
        if self.val is not None:
            # Already initialized
            return

        if not 0 < self.validation_split_ratio < 1:
            raise ValueError("validation_split_ratio must be in (0, 1)")

        # resolve all files so we can split them
        self.train.prepare_for_mode(PipelineMode.Training)
        params: FileDataParams = self.train
        n = int(self.validation_split_ratio * len(params.images))
        logger.info(f"Splitting training and validation files with ratio {self.validation_split_ratio}: "
                    f"{n}/{len(params.images) - n} for validation/training.")
        indices = list(range(len(params.images)))
        shuffle(indices)
        all_files = params.images
        all_text_files = params.texts

        # split train and val img/gt files. Use train settings
        params.images = [all_files[i] for i in indices[:n]]
        if all_text_files is not None:
            assert (len(all_text_files) == len(all_files))
            params.text_files = [all_text_files[i] for i in indices[:n]]

        self.val = params


@pai_dataclass
@dataclass
class DataParams(DataBaseParams):
    gen: TrainValGeneratorParamsBase = field(default_factory=CalamariDefaultTrainValGeneratorParams, metadata=pai_meta(
        choices=[CalamariDefaultTrainValGeneratorParams, CalamariSplitTrainValGeneratorParams, CalamariTrainOnlyGeneratorParams]
    ))
    skip_invalid_gt: bool = True
    input_channels: int = 1
    downscale_factor: int = field(default=-1, metadata=pai_meta(mode='ignore'))  # Set based on model
    line_height: int = field(default=48, metadata=pai_meta(help="The line height"))
    ensemble: int = field(default=0, metadata=pai_meta(mode='ignore'))  # Set based on model
    raw_dataset: bool = False
    codec: Optional[Codec] = field(default=None, metadata=pai_meta(mode='ignore'))

    def __post_init__(self):
        from calamari_ocr.ocr.dataset.imageprocessors.center_normalizer import CenterNormalizer
        from calamari_ocr.ocr.dataset.imageprocessors.scale_to_height_processor import ScaleToHeight
        for p in self.post_proc.processors + self.pre_proc.processors:
            if isinstance(p, ScaleToHeight):
                p.height = self.line_height
            elif isinstance(p, CenterNormalizer):
                p.line_height = self.line_height
