import json
from functools import partial

from tfaip.data.pipeline.definitions import Sample
from tfaip.predict.multimodelpredictor import MultiModelVoter

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
        self.datas = datas
        self.post_proc = post_proc
        self.out_to_in_transformer = out_to_in_transformer

    def vote(self, sample: Sample) -> Sample:
        inputs, outputs, meta = sample.inputs, sample.outputs, sample.meta
        prediction_results = []

        def out_to_in(x: int, meta) -> int:
            return self.out_to_in_transformer.local_to_global(
                x,
                model_factor=inputs["img_len"] / prediction.logits.shape[0],
                data_proc_params=meta,
            )

        for i, (prediction, m, data, post_) in enumerate(zip(outputs, meta, self.datas, self.post_proc)):
            prediction.id = f"fold_{i}"
            prediction_results.append(
                PredictionResult(
                    prediction,
                    codec=data.params.codec,
                    text_postproc=post_,
                    out_to_in_trans=partial(out_to_in, meta=m),
                )
            )
        # vote the results (if only one model is given, this will just return the sentences)
        prediction = self.voter.vote_prediction_result(prediction_results)
        prediction.id = "voted"
        return Sample(inputs=inputs, outputs=(prediction_results, prediction), meta=meta[0])
