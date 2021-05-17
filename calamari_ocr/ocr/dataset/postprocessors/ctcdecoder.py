from dataclasses import dataclass, field
from typing import Type

from paiargparse import pai_dataclass
from tfaip.data.pipeline.definitions import PipelineMode, Sample
from tfaip.data.pipeline.processor.dataprocessor import (
    MappingDataProcessor,
    DataProcessorParams,
)

from calamari_ocr.ocr.model.ctcdecoder.ctc_decoder import (
    create_ctc_decoder,
    CTCDecoderParams,
)


@pai_dataclass
@dataclass
class CTCDecoderProcessorParams(DataProcessorParams):
    ctc_decoder_params: CTCDecoderParams = field(default_factory=CTCDecoderParams)

    @staticmethod
    def cls() -> Type["MappingDataProcessor"]:
        return CTCDecoderProcessor


class CTCDecoderProcessor(MappingDataProcessor[CTCDecoderProcessorParams]):
    @staticmethod
    def default_params() -> dict:
        return {
            "ctc_decoder_params": CTCDecoderParams().to_dict(),
        }

    def __init__(
        self,
        params,
        data_params,
        mode: PipelineMode,
    ):
        super().__init__(params, data_params, mode)
        self.ctc_decoder = create_ctc_decoder(data_params.codec, self.params.ctc_decoder_params)

    def apply(self, sample: Sample) -> Sample:
        if sample.targets and "gt" in sample.targets:
            sample.targets["sentence"] = "".join(self.data_params.codec.decode(sample.targets["gt"]))
        if sample.outputs:

            def decode(suffix):
                outputs = self.ctc_decoder.decode(sample.outputs["softmax" + suffix].astype(float))
                outputs.labels = list(map(int, outputs.labels))
                outputs.sentence = "".join(self.data_params.codec.decode(outputs.labels))
                return outputs

            outputs = decode("")
            outputs.voter_predictions = []
            for i in range(self.data_params.ensemble):
                outputs.voter_predictions.append(decode(f"_{i}"))

            sample = sample.new_outputs(outputs)
        return sample
