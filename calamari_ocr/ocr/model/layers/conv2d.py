from dataclasses import dataclass, field
from typing import Type

from paiargparse import pai_dataclass, pai_meta
from tensorflow import keras

from calamari_ocr.ocr.model.layers.activation import activation_by_str
from calamari_ocr.ocr.model.layers.layer import LayerParams, IntVec2D, Layer


@pai_dataclass(alt="Conv")
@dataclass
class Conv2DLayerParams(LayerParams):
    @classmethod
    def name_prefix(cls) -> str:
        return "conv2d"

    @classmethod
    def cls(cls) -> Type["Layer"]:
        return Conv2DLayer

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


class Conv2DLayer(Layer[Conv2DLayerParams]):
    def __init__(self, *args, **kwargs):
        super(Conv2DLayer, self).__init__(*args, **kwargs)
        self.conv = keras.layers.Conv2D(
            name="conv",
            filters=self.params.filters,
            kernel_size=self.params.kernel_size.to_tuple(),
            strides=self.params.strides.to_tuple(),
            padding=self.params.padding,
            activation=activation_by_str(self.params.activation),
        )

    def input_dims(self) -> int:
        return 4

    def _call(self, inputs, **kwargs):
        return self.conv(inputs)
