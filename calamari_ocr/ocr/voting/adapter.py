from typing import List

from tfaip.data.pipeline.definitions import Sample
from tfaip.predict.multimodelvoter import MultiModelVoter

from calamari_ocr.ocr.predict.params import PredictionResult
from calamari_ocr.ocr.voting import voter_from_params
from calamari_ocr.utils.output_to_input_transformer import OutputToInputTransformer


class CalamariMultiModelVoter(MultiModelVoter):
    def __init__(
        self,
        voter_params,
        datas,
        post_proc,
        out_to_in_transformer: OutputToInputTransformer,
    ):
        self.voter = voter_from_params(voter_params)
        self.codecs = [d.params.codec for d in datas]
        self.out_to_in_transformer = out_to_in_transformer
        self.post_proc = post_proc

    def _convert_sample_to_prediction_results(self, sample: Sample) -> List[PredictionResult]:
        inputs, outputs, meta = sample.inputs, sample.outputs, sample.meta
        prediction_results = []

        for i, (prediction, m, codec, post_) in enumerate(zip(outputs, meta, self.codecs, self.post_proc)):
            prediction.id = f"fold_{i}"
            prediction_results.append(
                PredictionResult(
                    prediction,
                    codec=codec,
                    text_postproc=post_,
                    out_to_in_trans=make_out_to_in(
                        meta=m,
                        out_to_in_transformer=self.out_to_in_transformer,
                        model_factor=sample.inputs["img_len"] / prediction.logits.shape[0],
                    ),
                )
            )

        return prediction_results

    def vote(self, sample: Sample) -> Sample:
        inputs, outputs, meta = sample.inputs, sample.outputs, sample.meta
        prediction_results = self._convert_sample_to_prediction_results(sample)

        # vote the results (if only one model is given, this will just return the sentences)
        prediction = self.voter.vote_prediction_result(prediction_results)
        prediction.id = "voted"

        return Sample(inputs=inputs, outputs=(outputs, prediction), meta=meta)

    def finalize_sample(self, sample: Sample) -> Sample:
        prediction_results = self._convert_sample_to_prediction_results(sample.new_outputs(sample.outputs[0]))
        return sample.new_outputs((prediction_results, sample.outputs[1])).new_meta(sample.meta[0])


def make_out_to_in(meta, out_to_in_transformer, model_factor):
    def out_to_in(x: int) -> int:
        return out_to_in_transformer.local_to_global(
            x,
            model_factor=model_factor,
            data_proc_params=meta,
        )

    return out_to_in
