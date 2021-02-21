from dataclasses import dataclass
from typing import Type, List

from paiargparse import pai_dataclass
from tensorflow import keras

from calamari_ocr.ocr.model.layers.layer import LayerParams, Layer


@pai_dataclass
@dataclass
class ConcatLayerParams(LayerParams):
    @classmethod
    def cls(cls) -> Type['Layer']:
        return ConcatLayer

    concat_indices: List[int]


class ConcatLayer(Layer[ConcatLayerParams]):
    def __init__(self, *args, **kwargs):
        super(ConcatLayer, self).__init__(*args, **kwargs)

        self.concat = keras.layers.Concatenate()

    def input_dims(self) -> int:
        return -1  # arbitrary

    def _call(self, inputs, **kwargs):
        return self.concat([inputs[i] for i in self.params.concat_indices])
