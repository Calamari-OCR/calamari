from dataclasses import dataclass, field
from typing import Type

from paiargparse import pai_dataclass, pai_meta
from tensorflow import keras

from calamari_ocr.ocr.model.layers.layer import LayerParams, IntVec2D, Layer


@pai_dataclass(alt="DilatedBlock")
@dataclass
class DilatedBlockLayerParams(LayerParams):
    @classmethod
    def name_prefix(cls) -> str:
        return "dilated_block"

    @classmethod
    def cls(cls) -> Type["Layer"]:
        return DilatedBlockLayer

    def downscale(self, size: IntVec2D) -> IntVec2D:
        return IntVec2D(
            (size.x + self.strides.x - 1) // self.strides.x,
            (size.y + self.strides.y - 1) // self.strides.y,
        )

    def downscale_factor(self, factor: IntVec2D) -> IntVec2D:
        return IntVec2D(factor.x * self.strides.x, factor.y * self.strides.y)

    filters: int = 40
    kernel_size: IntVec2D = field(default_factory=lambda: IntVec2D(3, 3), metadata=pai_meta(tuple_like=True))
    strides: IntVec2D = field(default_factory=lambda: IntVec2D(1, 1), metadata=pai_meta(tuple_like=True))

    padding: str = "same"
    activation: str = "relu"
    dilated_depth: int = 2


class DilatedBlockLayer(Layer[DilatedBlockLayerParams]):
    def __init__(self, *args, **kwargs):
        super(DilatedBlockLayer, self).__init__(*args, **kwargs)
        depth = max(1, self.params.dilated_depth)
        assert self.params.filters % depth == 0
        self.dilated_layers = [
            keras.layers.Conv2D(
                name="conv2d_{}".format(i),
                filters=self.params.filters // depth,
                kernel_size=self.params.kernel_size.to_tuple(),
                strides=self.params.strides.to_tuple(),
                padding=self.params.padding,
                activation=self.params.activation,
                dilation_rate=2 ** (i + 1),
            )
            for i in range(depth)
        ]
        self.concat_layer = keras.layers.Concatenate(axis=-1)

    def input_dims(self) -> int:
        return 4

    def _call(self, inputs, **kwargs):
        ds = keras.backend.shape(inputs)
        ss = inputs.shape
        dilated_layers = [dl(inputs) for dl in self.dilated_layers]
        outputs = self.concat_layer(dilated_layers)
        outputs = keras.backend.reshape(
            outputs,
            [ds[0], ds[1], ss[2], sum([dl.filters for dl in self.dilated_layers])],
        )
        return outputs
