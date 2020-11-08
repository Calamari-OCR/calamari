from dataclasses import dataclass, field
from typing import List, Any

from dataclasses_json import dataclass_json
from tfaip.base.model import ModelBaseParams

from tfaip.base.trainer.trainer import TrainerParams as TfaipTrainerParams
from tfaip.util.enum import StrEnum

from calamari_ocr.ocr import Checkpoint


@dataclass_json
@dataclass
class IntVec2D:
    x: int = 0
    y: int = 0


@dataclass_json
@dataclass
class TrainerParams(TfaipTrainerParams):
    version: int = Checkpoint.VERSION

    skip_invalid_gt: bool = True
    stats_size: int = -1
    data_aug_retrain_on_original: bool = True  # Retrain the model with only the non augmented data in a second run
    current_stage_: int = -1  # Current training progress: 0 standard, 1 retraining on non aug.

    # TODO: early stopping
    # TODO: stats?
    codec_whitelist: List[str] = field(default_factory=list)
    keep_loaded_codec: bool = True
    preload_training: bool = True
    preload_validation: bool = True

    auto_compute_codec: bool = True
    progress_bar: bool = True

    auto_upgrade_checkpoints: bool = True



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


class CTCDecoderType(StrEnum):
    Default = 'default'
    TokenPassing = 'token_passing'
    WordBeamSearch = 'word_beam_search'


@dataclass_json
@dataclass
class CTCDecoder:
    type: CTCDecoderType
    blank_index: int
    min_p_threshold: float

    beam_width: int
    non_word_chars: List[str]
    dictionary: List[str]
    word_separator: str


@dataclass_json
@dataclass
class ModelParams(ModelBaseParams):
    layers: List[LayerParams] = field(default_factory=list)
    dropout: float = 0
    classes: int = -1
    ctc_merge_repeated: bool = True

    def compute_downscale_factor(self):
        factor = 1
        for layer in self.layers:
            if layer.type == LayerType.TransposedConv:
                factor //= layer.stride.x
            elif layer.type == LayerType.MaxPooling:
                factor *= layer.stride.x
        return factor


@dataclass_json
@dataclass
class Processor:
    type: str
    params: Any
    children: List['Processor'] = field(default_factory=list)


@dataclass_json
@dataclass
class LineGeneratorParams:
    fonts: List[str] = field(default_factory=list)
    font_size: int = 0
    min_script_offset: float = 0
    max_script_offset: float = 0


@dataclass_json
@dataclass
class TextGeneratorParams:
    word_length_mean: float = 0
    word_length_sigma: float = 0

    charset: List[str] = field(default_factory=list)
    super_charset: List[str] = field(default_factory=list)
    sub_charset: List[str] = field(default_factory=list)

    number_of_words_mean: float = 0
    number_of_words_sigma: float = 0
    word_separator: str = " "

    sub_script_p: float = 0
    super_script_p: float = 0
    bold_p: float = 0
    italic_p: float = 0

    letter_spacing_p: float = 0
    letter_spacing_mean: float = 0
    letter_spacing_sigma: float = 0
