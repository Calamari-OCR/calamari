import json
from functools import partial

from tensorflow import keras

from calamari_ocr.ocr.dataset.data import CalamariData
from calamari_ocr.ocr.voting import voter_from_params, VoterParams
from tfaip.base.data.pipeline.definitions import PipelineMode, InputOutputSample
from tfaip.base.device_config import DeviceConfig
from tfaip.base.predict import Predictor, PredictorParams, MultiModelPredictor
from tfaip.base.predict.multimodelpredictor import MultiModelVoter
from tqdm import tqdm

from typing import List, Callable

from calamari_ocr.ocr.dataset.params import CalamariPipelineParams
from calamari_ocr.ocr.scenario import CalamariScenario
from calamari_ocr.ocr import Codec, SavedModel
from calamari_ocr.utils.output_to_input_transformer import OutputToInputTransformer


class CalamariPredictorParams(PredictorParams):
    with_gt: bool = False
    ctc_decoder_params = None
    silent: bool = True


class PredictionResult:
    def __init__(self, prediction, codec, text_postproc, out_to_in_trans: Callable[[int], int], ground_truth=None):
        """ The output of a networks prediction (PredictionProto) with additional information

        It stores all required information for decoding (`codec`) and interpreting the output.

        Parameters
        ----------
        prediction : PredictionProto
            prediction the DNN
        codec : Codec
            codec required to decode the `prediction`
        text_postproc : TextPostprocessor
            text processor to apply to the decodec `prediction` to receive the actual prediction sentence
        """
        self.prediction = prediction
        self.logits = prediction.logits
        self.codec = codec
        self.text_postproc = text_postproc
        self.chars = codec.decode(prediction.labels)
        self.sentence = self.text_postproc.apply('', "".join(self.chars), {})[1]
        self.prediction.sentence = self.sentence
        self.out_to_in_trans = out_to_in_trans
        self.ground_truth = ground_truth

        self.prediction.avg_char_probability = 0

        for p in self.prediction.positions:
            for c in p.chars:
                c.char = codec.code2char[c.label]

            p.global_start = int(self.out_to_in_trans(p.local_start))
            p.global_end = int(self.out_to_in_trans(p.local_end))
            if len(p.chars) > 0:
                self.prediction.avg_char_probability += p.chars[0].probability

        self.prediction.avg_char_probability /= len(self.prediction.positions) if len(
            self.prediction.positions) > 0 else 1


class CalamariPredictor(Predictor):
    @staticmethod
    def from_checkpoint(params: CalamariPredictorParams, checkpoint: str, auto_update_checkpoints=True):
        ckpt = SavedModel(checkpoint, auto_update=False)
        trainer_params = CalamariScenario.trainer_params_from_dict(ckpt.json)
        trainer_params.scenario_params.data_params.pre_processors_.run_parallel = False
        trainer_params.scenario_params.data_params.post_processors_.run_parallel = False
        scenario = CalamariScenario(trainer_params.scenario_params)
        predictor = CalamariPredictor(params, scenario.create_data())
        ckpt = SavedModel(checkpoint, auto_update=auto_update_checkpoints)  # Device params must be specified first
        predictor.set_model(keras.models.load_model(ckpt.ckpt_path + '.h5', custom_objects=CalamariScenario.model_cls().get_all_custom_objects()))
        return predictor


class CalamariMultiModelVoter(MultiModelVoter):
    def __init__(self, voter_params, datas, post_proc, out_to_in_transformer):
        self.voter = voter_from_params(voter_params)
        self.datas = datas
        self.post_proc = post_proc
        self.out_to_in_transformer = out_to_in_transformer

    def vote(self, sample: InputOutputSample) -> InputOutputSample:
        inputs, outputs, meta = sample
        prediction_results = []
        input_meta = json.loads(inputs['meta'])
        for i, (prediction, m, data, post_) in enumerate(zip(outputs, meta, self.datas, self.post_proc)):
            prediction.id = "fold_{}".format(i)
            prediction_results.append(PredictionResult(prediction,
                                                       codec=data.params().codec,
                                                       text_postproc=post_,
                                                       out_to_in_trans=partial(self.out_to_in_transformer.local_to_global,
                                                                               model_factor=inputs['img_len'] /
                                                                                            prediction.logits.shape[0],
                                                                               data_proc_params=input_meta),
                                                       ))
        # vote the results (if only one model is given, this will just return the sentences)
        prediction = self.voter.vote_prediction_result(prediction_results)
        prediction.id = "voted"
        return InputOutputSample(inputs, (prediction_results, prediction), input_meta)


class CalamariMultiPredictor(MultiModelPredictor):
    @classmethod
    def from_paths(cls, checkpoints: List[str],
                   auto_update_checkpoints=True,
                   predictor_params: PredictorParams = None,
                   voter_params: VoterParams = None,
                   **kwargs
                   ) -> 'MultiModelPredictor':
        if not checkpoints:
            raise Exception("No checkpoints provided.")

        if predictor_params is None:
            predictor_params = PredictorParams(silent=True, progress_bar=True)

        DeviceConfig(predictor_params.device_params)
        checkpoints = [SavedModel(ckpt, auto_update=auto_update_checkpoints) for ckpt in checkpoints]
        multi_predictor = super(CalamariMultiPredictor, cls).from_paths(
            [ckpt.json_path for ckpt in checkpoints],
            predictor_params,
            CalamariScenario,
            model_paths=[ckpt.ckpt_path + '.h5' for ckpt in checkpoints],
            predictor_args={'voter_params': voter_params},
        )

        return multi_predictor

    def __init__(self, voter_params, *args, **kwargs):
        super(CalamariMultiPredictor, self).__init__(*args, **kwargs)
        self.voter_params = voter_params or VoterParams()

    def create_voter(self, data_params: 'DataBaseParams') -> MultiModelVoter:
        post_proc = [CalamariData.data_processor_factory().create_sequence(
            data.params().post_processors_.sample_processors[1:], data.params(), PipelineMode.Prediction) for
            data in self.datas]
        pre_proc = CalamariData.data_processor_factory().create_sequence(
            self.data.params().pre_processors_.sample_processors, self.data.params(),
            PipelineMode.Prediction)
        out_to_in_transformer = OutputToInputTransformer(pre_proc)
        return CalamariMultiModelVoter(self.voter_params, self.datas, post_proc, out_to_in_transformer)
