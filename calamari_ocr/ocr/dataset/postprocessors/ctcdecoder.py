from typing import Optional

from calamari_ocr.ocr.model.ctcdecoder.ctc_decoder import create_ctc_decoder, CTCDecoderParams
from tfaip.base.data.pipeline.dataprocessor import DataProcessor
from tfaip.base.data.pipeline.definitions import PipelineMode, Sample


class CTCDecoderProcessor(DataProcessor):
    @staticmethod
    def default_params() -> dict:
        return {
            'ctc_decoder_params': CTCDecoderParams().to_dict(),
        }

    def __init__(self,
                 params,
                 mode: PipelineMode,
                 ctc_decoder_params: Optional[CTCDecoderParams] = None,
                 ):
        super().__init__(params, mode)
        ctc_decoder_params = ctc_decoder_params or CTCDecoderParams
        if isinstance(ctc_decoder_params, dict):
            ctc_decoder_params = CTCDecoderParams.from_dict(ctc_decoder_params)

        self.ctc_decoder = create_ctc_decoder(params.codec, ctc_decoder_params)

    def apply(self, sample: Sample) -> Sample:
        if sample.targets and 'gt' in sample.targets:
            sample.targets['sentence'] = "".join(self.params.codec.decode(sample.targets['gt']))
        if sample.outputs:
            def decode(suffix):
                outputs = self.ctc_decoder.decode(sample.outputs['softmax' + suffix].astype(float))
                outputs.labels = list(map(int, outputs.labels))
                outputs.sentence = "".join(self.params.codec.decode(outputs.labels))
                return outputs

            outputs = decode("")
            outputs.voter_predictions = []
            for i in range(self.params.ensemble_):
                outputs.voter_predictions.append(decode(f"_{i}"))

            sample = sample.new_outputs(outputs)
        return sample
