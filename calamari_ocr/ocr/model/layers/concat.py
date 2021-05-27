from dataclasses import dataclass, field
from typing import Type, List

from paiargparse import pai_dataclass, pai_meta
from tensorflow import keras

from calamari_ocr.ocr.model.layers.layer import LayerParams, Layer


@pai_dataclass(alt="Concat")
@dataclass
class ConcatLayerParams(LayerParams):
    @classmethod
    def name_prefix(cls) -> str:
        return "concat"

    @classmethod
    def cls(cls) -> Type["Layer"]:
        return ConcatLayer

    concat_indices: List[int] = field(default_factory=list, metadata=pai_meta(required=True))


class ConcatLayer(Layer[ConcatLayerParams]):
    def __init__(self, *args, **kwargs):
        super(ConcatLayer, self).__init__(*args, **kwargs)

        self.concat = keras.layers.Concatenate()

    def input_dims(self) -> int:
        return -1  # arbitrary

    def _call(self, inputs, **kwargs):
        return self.concat([inputs[i] for i in self.params.concat_indices])
