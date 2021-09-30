from dataclasses import dataclass, field
from typing import Type

from paiargparse import pai_dataclass, pai_meta
from tensorflow import keras

from calamari_ocr.ocr.model.layers.layer import LayerParams, IntVec2D, Layer


@pai_dataclass(alt="Pool")
@dataclass
class MaxPool2DLayerParams(LayerParams):
    @classmethod
    def name_prefix(cls) -> str:
        return "maxpool2d"

    @classmethod
    def cls(cls) -> Type["Layer"]:
        return MaxPool2DLayer

    def downscale(self, size: IntVec2D) -> IntVec2D:
        strides = self.real_strides()
        return IntVec2D((size.x + strides.x - 1) // strides.x, (size.y + strides.y - 1) // strides.y)

    def downscale_factor(self, factor: IntVec2D) -> IntVec2D:
        strides = self.real_strides()
        return IntVec2D(factor.x * strides.x, factor.y * strides.y)

    def real_strides(self) -> IntVec2D:
        if self.strides is None:
            return self.pool_size
        return IntVec2D(
            self.strides.x if self.strides.x >= 0 else self.pool_size.x,
            self.strides.y if self.strides.y >= 0 else self.pool_size.y,
        )

    pool_size: IntVec2D = field(default_factory=lambda: IntVec2D(2, 2), metadata=pai_meta(tuple_like=True))
    strides: IntVec2D = field(default_factory=lambda: IntVec2D(-1, -1), metadata=pai_meta(tuple_like=True))

    padding: str = "same"


class MaxPool2DLayer(Layer[MaxPool2DLayerParams]):
    def __init__(self, *args, **kwargs):
        super(MaxPool2DLayer, self).__init__(*args, **kwargs)
        self.conv = keras.layers.MaxPool2D(
            name="conv",
            pool_size=self.params.pool_size.to_tuple(),
            strides=self.params.real_strides().to_tuple(),
            padding=self.params.padding,
        )

    def input_dims(self) -> int:
        return 4

    def _call(self, inputs, **kwargs):
        return self.conv(inputs)
