import re
from dataclasses import dataclass, field
from typing import List

from dataclasses_json import dataclass_json

from tfaip.base import TrainerParams as AIPTrainerParams

from calamari_ocr.ocr import SavedCalamariModel
from calamari_ocr.ocr.model.params import LayerParams, LayerType, LSTMDirection, ModelParams, IntVec2D


@dataclass_json
@dataclass
class TrainerParams(AIPTrainerParams):
    version: int = SavedCalamariModel.VERSION

    skip_invalid_gt: bool = True
    data_aug_retrain_on_original: bool = True  # Retrain the model with only the non augmented data in a second run
    current_stage: int = 0  # Current training progress: 0 standard, 1 retraining on non aug.

    codec_whitelist: List[str] = field(default_factory=list)
    keep_loaded_codec: bool = True
    preload_training: bool = True
    preload_validation: bool = True
    use_training_as_validation: bool = False

    auto_compute_codec: bool = True
    progress_bar: bool = True

    auto_upgrade_checkpoints: bool = True


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
