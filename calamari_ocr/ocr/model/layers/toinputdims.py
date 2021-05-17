from dataclasses import dataclass, field
from typing import Type

from paiargparse import pai_dataclass, pai_meta

from calamari_ocr.ocr.model.layers.layer import LayerParams, Layer


@pai_dataclass
@dataclass
class ToInputDimsLayerParams(LayerParams):
    @classmethod
    def name_prefix(cls) -> str:
        return "to_input_dims"

    @classmethod
    def cls(cls) -> Type["Layer"]:
        return ToInputDimsLayer

    dims: int = field(default=-1, metadata=pai_meta(required=True))


class ToInputDimsLayer(Layer[ToInputDimsLayerParams]):
    def input_dims(self) -> int:
        return self.params.dims

    def _call(self, inputs, **kwargs):
        return inputs
