from dataclasses import dataclass
from typing import Type

from paiargparse import pai_dataclass

from calamari_ocr.ocr.model.layers.layer import LayerParams, Layer


@pai_dataclass
@dataclass
class ToInputDimsLayerParams(LayerParams):
    @classmethod
    def cls(cls) -> Type['Layer']:
        return ToInputDimsLayer

    dims: int


class ToInputDimsLayer(Layer[ToInputDimsLayerParams]):
    def input_dims(self) -> int:
        return self.params.dims

    def _call(self, inputs, **kwargs):
        return inputs
