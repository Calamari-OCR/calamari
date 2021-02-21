from dataclasses import dataclass, field
from typing import Tuple, Type, Optional

from paiargparse import pai_dataclass
from tensorflow import keras

from calamari_ocr.ocr.model.layers.layer import LayerParams, IntVec2D, Layer


@pai_dataclass(alt="Pool")
@dataclass
class MaxPool2DLayerParams(LayerParams):
    @classmethod
    def name_prefix(cls) -> str:
        return 'maxpool2d'

    @classmethod
    def cls(cls) -> Type['Layer']:
        return MaxPool2DLayer

    def downscale(self, size: IntVec2D) -> IntVec2D:
        strides = self.strides if self.strides is not None else self.pool_size
        return IntVec2D((size.x + strides.x - 1) // strides.x, (size.y + strides.y - 1) // strides.y)

    def downscale_factor(self, factor: IntVec2D) -> IntVec2D:
        strides = self.strides if self.strides is not None else self.pool_size
        return IntVec2D(factor.x * strides.x, factor.y * strides.y)

    pool_size: IntVec2D = field(default_factory=lambda: IntVec2D(2, 2))
    strides: Optional[IntVec2D] = field(default=None)

    padding: str = "same"


class MaxPool2DLayer(Layer[MaxPool2DLayerParams]):
    def __init__(self, *args, **kwargs):
        super(MaxPool2DLayer, self).__init__(*args, **kwargs)
        self.conv = keras.layers.MaxPool2D(
            name="conv",
            pool_size=self.params.pool_size.to_tuple(),
            strides=self.params.strides.to_tuple() if self.params.strides else self.params.pool_size.to_tuple(),
            padding=self.params.padding,
        )

    def input_dims(self) -> int:
        return 4

    def _call(self, inputs, **kwargs):
        return self.conv(inputs)
