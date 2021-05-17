from dataclasses import dataclass
from typing import Type

from paiargparse import pai_dataclass
from tfaip.data.pipeline.definitions import Sample
from tfaip.data.pipeline.processor.dataprocessor import (
    MappingDataProcessor,
    DataProcessorParams,
)


@pai_dataclass
@dataclass
class ReshapeOutputsProcessorParams(DataProcessorParams):
    @staticmethod
    def cls() -> Type["MappingDataProcessor"]:
        return ReshapeOutputsProcessor


class ReshapeOutputsProcessor(MappingDataProcessor[ReshapeOutputsProcessorParams]):
    def apply(self, sample: Sample) -> Sample:
        inputs = sample.inputs
        outputs = sample.outputs
        assert inputs["img_len"].shape == (1,)
        inputs = inputs.copy()
        outputs = outputs.copy()
        inputs["img_len"] = inputs["img_len"][0]

        def reshape_outputs(suffix):
            out_len = "out_len" + suffix
            if out_len in outputs and outputs[out_len].shape == (1,):
                outputs[out_len] = outputs[out_len][0]

            for name in {
                "logits",
                "softmax",
                "blank_last_logits",
                "blank_last_softmax",
            }:
                name += suffix
                if name in outputs:
                    outputs[name] = outputs[name][: outputs[out_len]]

        reshape_outputs("")
        for i in range(self.data_params.ensemble):
            reshape_outputs(f"_{i}")

        return sample.new_inputs(inputs).new_outputs(outputs)
