from dataclasses import dataclass, field
from typing import List

from dataclasses_json import dataclass_json
from tfaip.base.model.modelbaseparams import ModelBaseParams

from tfaip.util.enum import StrEnum


@dataclass_json
@dataclass
class IntVec2D:
    x: int = 0
    y: int = 0


class LayerType(StrEnum):
    Convolutional = 'convolutional'
    MaxPooling = 'max_pooling'
    LSTM = 'lstm'
    TransposedConv = 'transposed_conv'
    DilatedBlock = 'dilated_block'
    Concat = 'concat'


class LSTMDirection(StrEnum):
    Bidirectional = 'bidirectional'


@dataclass_json
@dataclass
class LayerParams:
    type: LayerType

    # conv/pool
    filters: int = 0
    kernel_size: IntVec2D = field(default_factory=IntVec2D)
    stride: IntVec2D = field(default_factory=IntVec2D)

    # dilated block
    dilated_depth: int = 0

    # concat
    concat_indices: List[int] = field(default_factory=list)

    # lstm
    hidden_nodes: int = 0
    peepholes: bool = False
    lstm_direction: LSTMDirection = LSTMDirection.Bidirectional


@dataclass_json
@dataclass
class ModelParams(ModelBaseParams):
    layers: List[LayerParams] = field(default_factory=list)
    dropout: float = 0
    classes: int = -1
    ctc_merge_repeated: bool = True
    ensemble: int = 0  # For usage with the ensemble-model graph
    masking_mode: int = False  # This parameter is for evaluation only and should not be used in production

    def compute_downscale_factor(self):
        factor = 1
        for layer in self.layers:
            if layer.type == LayerType.TransposedConv:
                factor //= layer.stride.x
            elif layer.type == LayerType.MaxPooling:
                factor *= layer.stride.x
        return factor

    def compute_downscaled(self, length):
        for layer in self.layers:
            if layer.type == LayerType.TransposedConv:
                length = length * layer.stride.x
            elif layer.type == LayerType.MaxPooling:
                length = (length + layer.stride.x - 1) // layer.stride.x
        return length
