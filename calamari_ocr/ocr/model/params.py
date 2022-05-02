from dataclasses import dataclass, field
from typing import List, Tuple, Any, Union

from paiargparse import pai_dataclass, pai_meta
from tfaip.model.modelbaseparams import ModelBaseParams

from calamari_ocr.ocr.model.layers.layer import LayerParams, IntVec2D


def default_layers():
    from calamari_ocr.ocr.model.layers.conv2d import Conv2DLayerParams
    from calamari_ocr.ocr.model.layers.pool2d import MaxPool2DLayerParams
    from calamari_ocr.ocr.model.layers.bilstm import BiLSTMLayerParams
    from calamari_ocr.ocr.model.layers.dropout import DropoutLayerParams

    return [
        Conv2DLayerParams(filters=40),
        MaxPool2DLayerParams(),
        Conv2DLayerParams(filters=60),
        MaxPool2DLayerParams(),
        BiLSTMLayerParams(),
        DropoutLayerParams(rate=0.5),
    ]


def all_layers():
    from calamari_ocr.ocr.model.layers.bilstm import BiLSTMLayerParams
    from calamari_ocr.ocr.model.layers.concat import ConcatLayerParams
    from calamari_ocr.ocr.model.layers.conv2d import Conv2DLayerParams
    from calamari_ocr.ocr.model.layers.dilatedblock import DilatedBlockLayerParams
    from calamari_ocr.ocr.model.layers.dropout import DropoutLayerParams
    from calamari_ocr.ocr.model.layers.pool2d import MaxPool2DLayerParams
    from calamari_ocr.ocr.model.layers.transposedconv2d import TransposedConv2DLayerParams

    return [
        Conv2DLayerParams,
        ConcatLayerParams,
        MaxPool2DLayerParams,
        BiLSTMLayerParams,
        DropoutLayerParams,
        DilatedBlockLayerParams,
        TransposedConv2DLayerParams,
    ]


@pai_dataclass
@dataclass
class ModelParams(ModelBaseParams):
    layers: List[LayerParams] = field(
        default_factory=default_layers,
        metadata=pai_meta(choices=all_layers(), help="Layers of the graph. See the docs for more information."),
    )
    classes: int = -1
    ctc_merge_repeated: bool = True
    ensemble: int = 0  # For usage with the ensemble-model graph
    temperature: float = field(default=-1, metadata=pai_meta(help="Value to divide logits by (temperature scaling)."))
    masking_mode: int = False  # This parameter is for evaluation only and should not be used in production

    @staticmethod
    def cls():
        from calamari_ocr.ocr.model.model import Model

        return Model

    def graph_cls(self):
        from calamari_ocr.ocr.model.graph import CalamariGraph

        return CalamariGraph

    def __post_init__(self):
        # setup layer names
        counts = {}
        for layer in self.layers:
            counts[layer.name_prefix()] = counts.get(layer.name_prefix(), -1) + 1
            layer.name = f"{layer.name_prefix()}_{counts[layer.name_prefix()]}"

    def compute_downscale_factor(self) -> IntVec2D:
        factor = IntVec2D(1, 1)
        for layer in self.layers:
            factor = layer.downscale_factor(factor)
        return factor

    def compute_max_downscale_factor(self) -> IntVec2D:
        factor = IntVec2D(1, 1)
        max_factor = IntVec2D(1, 1)
        for layer in self.layers:
            factor = layer.downscale_factor(factor)
            max_factor.x = max(max_factor.x, factor.x)
            max_factor.y = max(max_factor.y, factor.y)
        return max_factor

    def compute_downscaled(self, size: Union[int, IntVec2D, Tuple[Any, Any]]):
        if isinstance(size, int):
            for layer in self.layers:
                size = layer.downscale(IntVec2D(size, 1)).x
        elif isinstance(size, IntVec2D):
            for layer in self.layers:
                size = layer.downscale(size)
        elif isinstance(size, tuple):
            for layer in self.layers:
                size = layer.downscale(IntVec2D(size[0], size[1]))
                size = size.x, size.y
        else:
            raise NotImplementedError
        return size
