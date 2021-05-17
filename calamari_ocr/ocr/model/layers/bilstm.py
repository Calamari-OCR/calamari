from dataclasses import dataclass
from typing import Type

from paiargparse import pai_dataclass
from tensorflow import keras

from calamari_ocr.ocr.model.layers.layer import LayerParams, Layer


@pai_dataclass(alt="BiLSTM")
@dataclass
class BiLSTMLayerParams(LayerParams):
    @classmethod
    def name_prefix(cls) -> str:
        return "lstm"

    @classmethod
    def cls(cls) -> Type["Layer"]:
        return BiLSTMLayer

    hidden_nodes: int = 200
    merge_mode: str = "concat"


class BiLSTMLayer(Layer[BiLSTMLayerParams]):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        lstm = keras.layers.LSTM(
            units=self.params.hidden_nodes,
            activation="tanh",
            recurrent_activation="sigmoid",
            recurrent_dropout=0,
            unroll=False,
            use_bias=True,
            return_sequences=True,
            unit_forget_bias=True,
            name="lstm",
        )
        self.lstm = keras.layers.Bidirectional(
            lstm,
            name="bidirectional",
            merge_mode=self.params.merge_mode,
        )

    def input_dims(self) -> int:
        return 3

    def _call(self, inputs, **kwargs):
        return self.lstm(inputs)
