from dataclasses import dataclass, field
from typing import Type

from paiargparse import pai_dataclass, pai_meta
from tensorflow import keras

from calamari_ocr.ocr.model.layers.layer import LayerParams, IntVec2D, Layer


@pai_dataclass(alt="TConv")
@dataclass
class TransposedConv2DLayerParams(LayerParams):
    @classmethod
    def name_prefix(cls) -> str:
        return "tconv2d"

    @classmethod
    def cls(cls) -> Type["Layer"]:
        return TransposedConv2DLayer

    def downscale(self, size: IntVec2D) -> IntVec2D:
        return IntVec2D(size.x * self.strides.x, size.y * self.strides.y)

    def downscale_factor(self, size: IntVec2D) -> IntVec2D:
        return IntVec2D(
            (size.x + self.strides.x - 1) // self.strides.x,
            (size.y + self.strides.y - 1) // self.strides.y,
        )

    filters: int = 40
    kernel_size: IntVec2D = field(default_factory=lambda: IntVec2D(3, 3), metadata=pai_meta(tuple_like=True))
    strides: IntVec2D = field(default_factory=lambda: IntVec2D(2, 2), metadata=pai_meta(tuple_like=True))

    padding: str = "same"
    activation: str = "relu"


class TransposedConv2DLayer(Layer[TransposedConv2DLayerParams]):
    def __init__(self, *args, **kwargs):
        super(TransposedConv2DLayer, self).__init__(*args, **kwargs)
        self.conv = keras.layers.Conv2DTranspose(
            name="conv",
            filters=self.params.filters,
            kernel_size=self.params.kernel_size.to_tuple(),
            strides=self.params.strides.to_tuple(),
            padding=self.params.padding,
            activation=self.params.activation,
        )

    def input_dims(self) -> int:
        return 4

    def _call(self, inputs, **kwargs):
        return self.conv(inputs)
