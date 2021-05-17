from copy import deepcopy
from dataclasses import dataclass, field
from random import shuffle
from typing import Optional
import logging

from paiargparse import pai_dataclass, pai_meta
from tfaip import TrainerPipelineParams, TrainerPipelineParamsBase, PipelineMode

from calamari_ocr.ocr.dataset.datareader.base import CalamariDataGeneratorParams
from calamari_ocr.ocr.dataset.datareader.file import FileDataParams
from calamari_ocr.ocr.dataset.datareader.pagexml.reader import PageXML
from calamari_ocr.ocr.dataset.params import DATA_GENERATOR_CHOICES


logger = logging.getLogger(__name__)


@pai_dataclass(alt="TrainVal")
@dataclass
class CalamariDefaultTrainerPipelineParams(
    TrainerPipelineParams[CalamariDataGeneratorParams, CalamariDataGeneratorParams]
):
    train: CalamariDataGeneratorParams = field(
        default_factory=FileDataParams,
        metadata=pai_meta(choices=DATA_GENERATOR_CHOICES, mode="flat"),
    )
    val: CalamariDataGeneratorParams = field(
        default_factory=FileDataParams,
        metadata=pai_meta(choices=DATA_GENERATOR_CHOICES, mode="flat"),
    )


@pai_dataclass(alt="TrainOnly")
@dataclass
class CalamariTrainOnlyPipelineParams(
    TrainerPipelineParamsBase[CalamariDataGeneratorParams, CalamariDataGeneratorParams]
):
    def train_gen(self) -> CalamariDataGeneratorParams:
        return self.train

    def val_gen(self) -> Optional[CalamariDataGeneratorParams]:
        return None

    train: CalamariDataGeneratorParams = field(
        default_factory=FileDataParams,
        metadata=pai_meta(choices=DATA_GENERATOR_CHOICES, mode="flat"),
    )


@pai_dataclass(alt="SplitTrain")
@dataclass
class CalamariSplitTrainerPipelineParams(
    TrainerPipelineParams[CalamariDataGeneratorParams, CalamariDataGeneratorParams]
):
    train: CalamariDataGeneratorParams = field(
        default_factory=FileDataParams,
        metadata=pai_meta(
            choices=[FileDataParams, PageXML],
            enforce_choices=True,
            mode="flat",
        ),
    )
    validation_split_ratio: float = field(
        default=0.2,
        metadata=pai_meta(help="Use factor of n of the training dataset for validation."),
    )

    val: Optional[CalamariDataGeneratorParams] = field(default=None, metadata=pai_meta(mode="ignore"))

    def __post_init__(self):
        if self.val is not None:
            # Already initialized
            return

        if not 0 < self.validation_split_ratio < 1:
            raise ValueError("validation_split_ratio must be in (0, 1)")

        # resolve all files so we can split them
        self.train.prepare_for_mode(PipelineMode.TRAINING)
        self.val = deepcopy(self.train)
        samples = len(self.train)
        n = int(self.validation_split_ratio * samples)
        if n == 0:
            raise ValueError(
                f"Ratio is to small since {self.validation_split_ratio} * {samples} = {n}. "
                f"Increase the amount of data or the split ratio."
            )
        logger.info(
            f"Splitting training and validation files with ratio {self.validation_split_ratio}: "
            f"{n}/{samples - n} for validation/training."
        )
        indices = list(range(samples))
        shuffle(indices)

        # split train and val img/gt files. Use train settings
        self.train.select(indices[n:])
        self.val.select(indices[:n])
