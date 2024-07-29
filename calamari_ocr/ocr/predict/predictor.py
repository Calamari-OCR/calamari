from typing import List

from tensorflow import keras

from tfaip.data.pipeline.processor.params import SequentialProcessorPipelineParams
from tfaip.device.device_config import DeviceConfig
import tfaip.imports as tfaip_cls
from tfaip.predict.multimodelpredictor import MultiModelVoter

from calamari_ocr.ocr.predict.params import PredictorParams
from calamari_ocr.ocr.scenario import CalamariScenario
from calamari_ocr.ocr.voting import VoterParams
from calamari_ocr.ocr import SavedCalamariModel, DataParams
from calamari_ocr.ocr.voting.adapter import CalamariMultiModelVoter
from calamari_ocr.utils.output_to_input_transformer import OutputToInputTransformer


class Predictor(tfaip_cls.Predictor):
    @staticmethod
    def from_checkpoint(params: PredictorParams, checkpoint: str, auto_update_checkpoints=True):
        DeviceConfig(params.device)  # Device must be specified first
        ckpt = SavedCalamariModel(checkpoint, auto_update=auto_update_checkpoints)
        scenario_params = CalamariScenario.params_from_dict(ckpt.dict)
        scenario = CalamariScenario(scenario_params)
        predictor = Predictor(params, scenario.create_data())
        predictor.set_model(
            keras.models.load_model(
                ckpt.ckpt_path,
                custom_objects=CalamariScenario.model_cls().all_custom_objects(),
            )
        )
        return predictor


class MultiPredictor(tfaip_cls.MultiModelPredictor):
    @classmethod
    def from_paths(
        cls,
        checkpoints: List[str],
        auto_update_checkpoints=True,
        predictor_params: PredictorParams = None,
        voter_params: VoterParams = None,
        **kwargs,
    ) -> "tfaip_cls.MultiModelPredictor":
        if not checkpoints:
            raise Exception("No checkpoints provided.")

        if predictor_params is None:
            predictor_params = PredictorParams(silent=True, progress_bar=True)

        DeviceConfig(predictor_params.device)
        checkpoints = [SavedCalamariModel(ckpt, auto_update=auto_update_checkpoints) for ckpt in checkpoints]

        multi_predictor = super(MultiPredictor, cls).from_paths(
            [ckpt.json_path for ckpt in checkpoints],
            predictor_params,
            CalamariScenario,
            model_paths=[ckpt.ckpt_path for ckpt in checkpoints],
            predictor_args={"voter_params": voter_params},
        )

        return multi_predictor

    def __init__(self, voter_params, *args, **kwargs):
        super(MultiPredictor, self).__init__(*args, **kwargs)
        self.voter_params = voter_params or VoterParams()

    def create_voter(self, data_params: "DataParams") -> MultiModelVoter:
        # Cut non text processors (first two)
        # force run_parallel = False because the voter itself already runs in separate threads
        post_proc_params = [
            SequentialProcessorPipelineParams(run_parallel=False, processors=data.params.post_proc.processors[2:])
            for data in self.datas
        ]
        post_proc = [p.create(self.params.pipeline, self.data.params) for p in post_proc_params]
        pre_proc = self.data.params.pre_proc.create(self.params.pipeline, self.data.params)
        out_to_in_transformer = OutputToInputTransformer(pre_proc)
        return CalamariMultiModelVoter(self.voter_params, self.datas, post_proc, out_to_in_transformer)
