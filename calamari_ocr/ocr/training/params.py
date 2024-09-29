import json
import logging
import re
from dataclasses import dataclass, field
from typing import Optional, List

from paiargparse import pai_meta, pai_dataclass
from tfaip import TrainerParams as AIPTrainerParams, TrainerPipelineParamsBase
from tfaip.util.tfaipargparse import post_init

from calamari_ocr.utils import resource_filename
from calamari_ocr.ocr import SavedCalamariModel
from calamari_ocr.ocr.dataset.codec import CodecConstructionParams
from calamari_ocr.ocr.model.layers.bilstm import BiLSTMLayerParams
from calamari_ocr.ocr.model.layers.concat import ConcatLayerParams
from calamari_ocr.ocr.model.layers.conv2d import Conv2DLayerParams
from calamari_ocr.ocr.model.layers.dilatedblock import DilatedBlockLayerParams
from calamari_ocr.ocr.model.layers.dropout import DropoutLayerParams
from calamari_ocr.ocr.model.layers.pool2d import MaxPool2DLayerParams
from calamari_ocr.ocr.model.layers.transposedconv2d import TransposedConv2DLayerParams
from calamari_ocr.ocr.model.params import LayerParams, IntVec2D
from calamari_ocr.ocr.scenario import CalamariScenarioParams
from calamari_ocr.ocr.training.pipeline_params import (
    CalamariDefaultTrainerPipelineParams,
    CalamariTrainOnlyPipelineParams,
    CalamariSplitTrainerPipelineParams,
)

logger = logging.getLogger(__name__)


@pai_dataclass
@dataclass
class TrainerParams(AIPTrainerParams[CalamariScenarioParams, CalamariDefaultTrainerPipelineParams]):
    version: int = SavedCalamariModel.VERSION

    data_aug_retrain_on_original: bool = field(
        default=True,
        metadata=pai_meta(
            help="When training with augmentations usually the model is retrained in a second run with "
            "only the non augmented data. This will take longer. Use this flag to disable this "
            "behavior."
        ),
    )

    # Current training progress: 0 standard, 1 retraining on non aug.
    current_stage: int = field(default=0, metadata=pai_meta(mode="ignore"))
    scale_epoch_size_no_da_train: float = field(
        default=-1,
        metadata=pai_meta(
            help="Similal to --scale_epoch_size but for no data augmentation training (stage=1), "
            "defaults to --scale_epoch_size"
        ),
    )

    progress_bar: bool = True

    auto_upgrade_checkpoints: bool = field(
        default=True,
        metadata=pai_meta(help="Automatically update older checkpoints for warm start."),
    )

    codec: CodecConstructionParams = field(
        default_factory=CodecConstructionParams,
        metadata=pai_meta(
            help="Parameters defining how to construct the codec.",
            mode="flat",  # The actual codec is stored in data
        ),
    )

    gen: TrainerPipelineParamsBase = field(
        default_factory=CalamariDefaultTrainerPipelineParams,
        metadata=pai_meta(
            help="Parameters that setup the data generators (i.e. the input data).",
            disable_subclass_check=False,
            choices=[
                CalamariDefaultTrainerPipelineParams,
                CalamariTrainOnlyPipelineParams,
                CalamariSplitTrainerPipelineParams,
            ],
        ),
    )

    best_model_prefix: str = field(
        default="best",
        metadata=pai_meta(help="The prefix of the best model using early stopping"),
    )

    network: Optional[str] = field(
        default=None,
        metadata=pai_meta(
            mode="flat",
            help="Pass a network configuration to construct a simple graph. "
            "Alternatively, you can pass a predefined network name or a path to a json file."
            "Defaults to: --network=cnn=40:3x3,pool=2x2,cnn=60:3x3,pool=2x2,lstm=200,dropout=0.5",
        ),
    )

    def __post_init__(self):
        self.scenario.default_serve_dir = f"{self.best_model_prefix}.ckpt"
        self.scenario.trainer_params_filename = f"{self.best_model_prefix}.ckpt.json"
        self.early_stopping.best_model_name = ""

        self.gen.train_gen().n_folds = self.scenario.model.ensemble
        self.gen.train_gen().channels = self.scenario.data.input_channels
        if self.gen.val_gen() is not None:
            self.gen.val_gen().channels = self.scenario.data.input_channels
            self.gen.val_gen().n_folds = self.scenario.model.ensemble

        if self.network:
            self.scenario.model.layers = parse_network_param(self.network)
            post_init(self.scenario.model)


def set_default_networkparams(params: TrainerParams):
    params.optimizer_params.optimizer = "Adam"
    params.scenario_params.model_params.dropout = 0
    params.scenario_params.model_params.ctc_merge_repeated = True
    params.learning_rate_params.lr = 1e-3


def parse_network_param(s: str) -> List[LayerParams]:
    if s.endswith(".json"):
        # parse as jsonfile
        with open(s) as f:
            d = json.load(f)
            return list(LayerParams.from_dict(layer_params) for layer_params in d)

    if s in _default_networks:
        # parse default networks
        with open(_default_networks[s]) as f:
            d = json.load(f)
            return list(LayerParams.from_dict(layer_params) for layer_params in d)

    return graph_params_from_definition_string(s)


def graph_params_from_definition_string(s: str) -> List[LayerParams]:
    layers = []
    cnn_matcher = re.compile(r"^([\d]+)(:([\d]+)x([\d]+))?(:([\d]+)x([\d]+))?$")
    db_matcher = re.compile(r"^([\d]+):([\d]+)(:([\d]+)(x([\d]+))?)?$")
    concat_matcher = re.compile(r"^([\-\d]+):([\-\d]+)$")
    pool_matcher = re.compile(r"^([\d]+)(x([\d]+))?(:([\d]+)x([\d]+))?$")
    str_params = s.split(",")
    lstm_appeared = False
    for param in str_params:
        label, value = tuple(param.split("="))
        if label == "dropout":
            layers.append(DropoutLayerParams(rate=float(value)))
        elif label == "lstm":
            lstm_appeared = True
            layers.append(BiLSTMLayerParams(hidden_nodes=int(value)))
        elif label == "concat":
            if lstm_appeared:
                raise Exception("LSTM layers must be placed proceeding to CNN/Pool")

            match = concat_matcher.match(value)
            if match is None:
                raise Exception("Concat structure needs: concat=[index0]:[index1] but got concat={}".format(value))

            match = match.groups()
            layers.append(ConcatLayerParams(concat_indices=list(map(int, match))))
        elif label == "db":
            if lstm_appeared:
                raise Exception("LSTM layers must be placed proceeding to CNN/Pool")

            match = db_matcher.match(value)
            if match is None:
                raise Exception("Dilated block structure needs: db=[filters]:[depth>0]:[h]x[w]")

            match = match.groups()
            kernel_size = [3, 3]
            if match[2] is not None:
                kernel_size = [int(match[3])] * 2
            if match[4] is not None:
                kernel_size = [int(match[3]), int(match[5])]

            layers.append(
                DilatedBlockLayerParams(
                    filters=int(match[0]),
                    dilated_depth=int(match[1]),
                    kernel_size=IntVec2D(*kernel_size),
                    strides=IntVec2D(1, 1),
                )
            )
        elif label in {"cnn", "conv", "conv2d"}:
            if lstm_appeared:
                raise Exception("LSTM layers must be placed proceeding to CNN/Pool")

            match = cnn_matcher.match(value)
            if match is None:
                raise Exception(f"CNN structure needs: cnn=[filters]:[h]x[w]:[sx]x[sy] but got {value}")

            match = match.groups()
            kernel_size = [3, 3]
            stride = [1, 1]
            if match[1] is not None:
                kernel_size = [int(match[2]), int(match[3])]
            if match[4] is not None:
                stride = [int(match[5]), int(match[6])]

            layers.append(
                Conv2DLayerParams(
                    filters=int(match[0]),
                    kernel_size=IntVec2D(*kernel_size),
                    strides=IntVec2D(*stride),
                )
            )
        elif label in {"tcnn", "tconv", "tconv2d"}:
            if lstm_appeared:
                raise Exception("LSTM layers must be placed proceeding to CNN/Pool")

            match = cnn_matcher.match(value)
            if match is None:
                raise Exception("Transposed CNN structure needs: tcnn=[filters]:[sx]x[sy]")

            match = match.groups()
            kernel_size = [3, 3]
            stride = [2, 2]
            if match[1] is not None:
                stride = [int(match[2]), int(match[3])]
            if match[4] is not None:
                kernel_size = [int(match[5]), int(match[6])]

            layers.append(
                TransposedConv2DLayerParams(
                    filters=int(match[0]),
                    kernel_size=IntVec2D(*kernel_size),
                    strides=IntVec2D(*stride),
                )
            )
        elif label in {"pool", "max_pool", "pool2d"}:
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
                stride = IntVec2D(int(match[4]), int(match[5]))
            else:
                stride = IntVec2D(-1, -1)

            layers.append(MaxPool2DLayerParams(pool_size=IntVec2D(*kernel_size), strides=stride))
        else:
            raise Exception("Unknown layer with name: {}".format(label))

    return layers


# LOAD DEFAULT NETWORK ARCHITECTURES
# =================================

_networks_dir = resource_filename("calamari_ocr", "resources") / "networks"
_default_networks = {p.stem: p for p in _networks_dir.iterdir() if p.suffix == ".json"}

logger.debug(f"Found default networks: {list(_default_networks.keys())}")
