from dataclasses import dataclass
from typing import Type

from paiargparse import pai_dataclass
from tensorflow import keras

from calamari_ocr.ocr.model.layers.layer import LayerParams, Layer


@pai_dataclass(alt="Dropout")
@dataclass
class DropoutLayerParams(LayerParams):
    @classmethod
    def name_prefix(cls) -> str:
        return "dropout"

    @classmethod
    def cls(cls) -> Type["Layer"]:
        return DropoutLayer

    rate: float = 0.5


class DropoutLayer(Layer[DropoutLayerParams]):
    def __init__(self, *args, **kwargs):
        super(DropoutLayer, self).__init__(*args, **kwargs)
        self.dropout = keras.layers.Dropout(rate=self.params.rate)

    def input_dims(self) -> int:
        return -1  # arbitrary

    def _call(self, inputs, **kwargs):
        return self.dropout(inputs)
