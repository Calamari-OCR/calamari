from dataclasses import dataclass
from typing import Any

from dataclasses_json import dataclass_json

import numpy as np

from tfaip.base.data.pipeline.dataprocessor import DataProcessor
from tfaip.base.data.pipeline.definitions import PipelineMode


@dataclass_json
@dataclass
class NetworkPredictionResult:
    softmax: np.array
    output_length: int
    decoded: np.array
    params: Any = None
    ground_truth: np.array = None


class PredictionResultProcessor(DataProcessor):
    def __init__(self,
                 params,
                 mode: PipelineMode,
                 ):
        super().__init__(params, mode)

    def apply(self, inputs, outputs, meta: dict):
        return inputs, pred
