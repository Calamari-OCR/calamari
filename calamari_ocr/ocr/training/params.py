import logging
import re
from copy import deepcopy
from dataclasses import dataclass, field
from random import shuffle
from typing import Optional

from paiargparse import pai_meta, pai_dataclass
from tfaip.base import TrainerParams as AIPTrainerParams, TrainerPipelineParams, TrainerPipelineParamsBase, \
    PipelineMode

from calamari_ocr.ocr import SavedCalamariModel
from calamari_ocr.ocr.dataset.codec import CodecConstructionParams
from calamari_ocr.ocr.dataset.datareader.base import CalamariDataGeneratorParams
from calamari_ocr.ocr.dataset.datareader.file import FileDataParams
from calamari_ocr.ocr.dataset.datareader.pagexml.reader import PageXML
from calamari_ocr.ocr.dataset.params import DATA_GENERATOR_CHOICES
from calamari_ocr.ocr.model.params import LayerParams, LayerType, LSTMDirection, ModelParams, IntVec2D
from calamari_ocr.ocr.scenario import ScenarioParams

logger = logging.getLogger(__name__)


@pai_dataclass
@dataclass
class CalamariDefaultTrainerPipelineParams(TrainerPipelineParams[CalamariDataGeneratorParams,
                                                                 CalamariDataGeneratorParams]):
    train: CalamariDataGeneratorParams = field(default_factory=FileDataParams,
                                               metadata=pai_meta(choices=DATA_GENERATOR_CHOICES, mode='flat'))
    val: CalamariDataGeneratorParams = field(default_factory=FileDataParams,
                                             metadata=pai_meta(choices=DATA_GENERATOR_CHOICES, mode='flat'))


@pai_dataclass
@dataclass
class CalamariTrainOnlyPipelineParams(
    TrainerPipelineParamsBase[CalamariDataGeneratorParams, CalamariDataGeneratorParams]):
    def train_gen(self) -> CalamariDataGeneratorParams:
        return self.train

    def val_gen(self) -> Optional[CalamariDataGeneratorParams]:
        return None

    train: CalamariDataGeneratorParams = field(default_factory=FileDataParams,
                                               metadata=pai_meta(choices=DATA_GENERATOR_CHOICES, mode='flat'))


@pai_dataclass
@dataclass
class CalamariSplitTrainerPipelineParams(TrainerPipelineParams[CalamariDataGeneratorParams,
                                                               CalamariDataGeneratorParams]):
    train: CalamariDataGeneratorParams = field(default_factory=FileDataParams, metadata=pai_meta(
        choices=[FileDataParams, PageXML], enforce_choices=True, mode='flat',
    ))
    validation_split_ratio: float = field(default=0.2, metadata=pai_meta(
        help="Use factor of n of the training dataset for validation."))

    val: Optional[CalamariDataGeneratorParams] = field(default=None, metadata=pai_meta(mode="ignore"))

    def __post_init__(self):
        if self.val is not None:
            # Already initialized
            return

        if not 0 < self.validation_split_ratio < 1:
            raise ValueError("validation_split_ratio must be in (0, 1)")

        # resolve all files so we can split them
        self.train.prepare_for_mode(PipelineMode.Training)
        self.val = deepcopy(self.train)
        samples = len(self.train)
        n = int(self.validation_split_ratio * samples)
        if n == 0:
            raise ValueError(f"Ratio is to small since {self.validation_split_ratio} * {samples} = {n}. "
                             f"Increase the amount of data or the split ratio.")
        logger.info(f"Splitting training and validation files with ratio {self.validation_split_ratio}: "
                    f"{n}/{samples - n} for validation/training.")
        indices = list(range(samples))
        shuffle(indices)

        # split train and val img/gt files. Use train settings
        self.train.select(indices[n:])
        self.val.select(indices[:n])


@pai_dataclass
@dataclass
class TrainerParams(AIPTrainerParams[ScenarioParams, CalamariDefaultTrainerPipelineParams]):
    version: int = SavedCalamariModel.VERSION

    data_aug_retrain_on_original: bool = field(default=True, metadata=pai_meta(
        help="When training with augmentations usually the model is retrained in a second run with "
             "only the non augmented data. This will take longer. Use this flag to disable this "
             "behavior."))
    current_stage: int = 0  # Current training progress: 0 standard, 1 retraining on non aug.

    progress_bar: bool = True

    auto_upgrade_checkpoints: bool = True

    codec: CodecConstructionParams = field(default_factory=CodecConstructionParams, metadata=pai_meta(
        help="Parameters defining how to construct the codec.", mode='flat'  # The actual codec is stored in data
    ))

    best_model_prefix: str = field(default="best", metadata=pai_meta(
        help="The prefix of the best model using early stopping"))

    def __post_init__(self):
        self.scenario.default_serve_dir = f'{self.best_model_prefix}.ckpt.h5'
        self.scenario.trainer_params_filename = f'{self.best_model_prefix}.ckpt.json'
        self.early_stopping.best_model_name = ''

        self.gen.train_gen().n_folds = self.scenario.model.ensemble
        if self.gen.val_gen():
            self.gen.val_gen().n_folds = self.scenario.model.ensemble


def set_default_network_params(params: TrainerParams):
    params.optimizer_params.optimizer = 'Adam'
    params.scenario_params.model_params.dropout = 0
    params.scenario_params.model_params.ctc_merge_repeated = True
    params.learning_rate_params.lr = 1e-3


def params_from_definition_string(s: str, trainer_params: TrainerParams):
    model_params: ModelParams = trainer_params.scenario_params.model_params
    cnn_matcher = re.compile(r"^([\d]+)(:([\d]+)(x([\d]+))?)?$")
    db_matcher = re.compile(r"^([\d]+):([\d]+)(:([\d]+)(x([\d]+))?)?$")
    concat_matcher = re.compile(r"^([\-\d]+):([\-\d]+)$")
    pool_matcher = re.compile(r"^([\d]+)(x([\d]+))?(:([\d]+)x([\d]+))?$")
    str_params = s.split(",")
    lstm_appeared = False
    set_default_network_params(trainer_params)
    for param in str_params:
        label, value = tuple(param.split("="))
        model_flags = ["ctc_merge_repeated"]
        if label in model_flags:
            setattr(trainer_params.scenario_params.model_params, label, value.lower() == "true")
        elif label == "l_rate" or label == 'learning_rate':
            trainer_params.learning_rate_params.lr = float(value)
        elif label == "momentum":
            trainer_params.optimizer_params.momentum = float(value)
        elif label == 'dropout':
            trainer_params.scenario_params.model_params.dropout = float(value)
        elif label == "solver":
            trainer_params.optimizer_params.optimizer = value
        elif label == "lstm":
            lstm_appeared = True
            model_params.layers.append(LayerParams(
                type=LayerType.LSTM,
                lstm_direction=LSTMDirection.Bidirectional,
                hidden_nodes=int(value)
            ))
        elif label == 'concat':
            if lstm_appeared:
                raise Exception("LSTM layers must be placed proceeding to CNN/Pool")

            match = concat_matcher.match(value)
            if match is None:
                raise Exception("Concat structure needs: concat=[index0]:[index1] but got concat={}".format(value))

            match = match.groups()
            model_params.layers.append(
                LayerParams(
                    type=LayerType.Concat,
                    concat_indices=list(map(int, match))
                )
            )
        elif label == "db":
            if lstm_appeared:
                raise Exception("LSTM layers must be placed proceeding to CNN/Pool")

            match = db_matcher.match(value)
            if match is None:
                raise Exception("Dilated block structure needs: db=[filters]:[depth>0]:[h]x[w]")

            match = match.groups()
            kernel_size = [2, 2]
            if match[2] is not None:
                kernel_size = [int(match[3])] * 2
            if match[4] is not None:
                kernel_size = [int(match[3]), int(match[5])]

            model_params.layers.append(LayerParams(
                type=LayerType.DilatedBlock,
                filters=int(match[0]),
                dilated_depth=int(match[1]),
                kernel_size=IntVec2D(*kernel_size),
                stride=IntVec2D(1, 1),
            ))
        elif label == "cnn":
            if lstm_appeared:
                raise Exception("LSTM layers must be placed proceeding to CNN/Pool")

            match = cnn_matcher.match(value)
            if match is None:
                raise Exception("CNN structure needs: cnn=[filters]:[h]x[w] but got {}".format(value))

            match = match.groups()
            kernel_size = [2, 2]
            if match[1] is not None:
                kernel_size = [int(match[2])] * 2
            if match[3] is not None:
                kernel_size = [int(match[2]), int(match[4])]

            model_params.layers.append(LayerParams(
                type=LayerType.Convolutional,
                filters=int(match[0]),
                kernel_size=IntVec2D(*kernel_size),
                stride=IntVec2D(1, 1),
            ))
        elif label == "tcnn":
            if lstm_appeared:
                raise Exception("LSTM layers must be placed proceeding to CNN/Pool")

            match = cnn_matcher.match(value)
            if match is None:
                raise Exception("Transposed CNN structure needs: tcnn=[filters]:[sx]x[sy]")

            match = match.groups()
            kernel_size = [2, 2]
            stride = [2, 2]
            if match[1] is not None:
                stride = [int(match[2])] * 2
            if match[3] is not None:
                stride = [int(match[2]), int(match[4])]

            model_params.layers.append(LayerParams(
                type=LayerType.TransposedConv,
                filters=int(match[0]),
                kernel_size=IntVec2D(*kernel_size),
                stride=IntVec2D(*stride),
            ))
        elif label == "pool":
            if lstm_appeared:
                raise Exception("LSTM layers must be placed proceeding to CNN/Pool")
            match = pool_matcher.match(value)
            if match is None:
                raise Exception("Pool structure needs: pool=[h];[w]")

            match = match.groups()
            kernel_size = [int(match[0])] * 2
            if match[1] is not None:
                kernel_size = [int(match[0]), int(match[2])]

            if match[3] is not None:
                stride = [int(match[4]), int(match[5])]
            else:
                stride = kernel_size

            model_params.layers.append(LayerParams(
                type=LayerType.MaxPooling,
                kernel_size=IntVec2D(*kernel_size),
                stride=IntVec2D(*stride),
            ))
        else:
            raise Exception("Unknown layer with name: {}".format(label))
