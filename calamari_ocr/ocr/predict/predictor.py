from typing import List

from tensorflow import keras

from tfaip.base.data.pipeline.definitions import PipelineMode
from tfaip.base.device_config import DeviceConfig
import tfaip.base.predict as aip_predict
from tfaip.base.predict.multimodelpredictor import MultiModelVoter

from calamari_ocr.ocr.predict.params import PredictorParams
from calamari_ocr.ocr.scenario import Scenario
from calamari_ocr.ocr.dataset.data import Data
from calamari_ocr.ocr.voting import VoterParams
from calamari_ocr.ocr import SavedCalamariModel, DataParams
from calamari_ocr.ocr.voting.adapter import CalamariMultiModelVoter
from calamari_ocr.utils.output_to_input_transformer import OutputToInputTransformer


class Predictor(aip_predict.Predictor):
    @staticmethod
    def from_checkpoint(params: PredictorParams, checkpoint: str, auto_update_checkpoints=True):
        ckpt = SavedCalamariModel(checkpoint, auto_update=False)
        trainer_params = Scenario.trainer_params_from_dict(ckpt.dict)
        trainer_params.scenario_params.data_params.pre_processors_.run_parallel = False
        trainer_params.scenario_params.data_params.post_processors_.run_parallel = False
        scenario = Scenario(trainer_params.scenario_params)
        predictor = Predictor(params, scenario.create_data())
        ckpt = SavedCalamariModel(checkpoint, auto_update=auto_update_checkpoints)  # Device params must be specified first
        predictor.set_model(keras.models.load_model(ckpt.ckpt_path + '.h5', custom_objects=Scenario.model_cls().get_all_custom_objects()))
        return predictor


class MultiPredictor(aip_predict.MultiModelPredictor):
    @classmethod
    def from_paths(cls, checkpoints: List[str],
                   auto_update_checkpoints=True,
                   predictor_params: PredictorParams = None,
                   voter_params: VoterParams = None,
                   **kwargs
                   ) -> 'aip_predict.MultiModelPredictor':
        if not checkpoints:
            raise Exception("No checkpoints provided.")

        if predictor_params is None:
            predictor_params = PredictorParams(silent=True, progress_bar=True)

        DeviceConfig(predictor_params.device_params)
        checkpoints = [SavedCalamariModel(ckpt, auto_update=auto_update_checkpoints) for ckpt in checkpoints]
        multi_predictor = super(MultiPredictor, cls).from_paths(
            [ckpt.json_path for ckpt in checkpoints],
            predictor_params,
            Scenario,
            model_paths=[ckpt.ckpt_path + '.h5' for ckpt in checkpoints],
            predictor_args={'voter_params': voter_params},
        )

        return multi_predictor

    def __init__(self, voter_params, *args, **kwargs):
        super(MultiPredictor, self).__init__(*args, **kwargs)
        self.voter_params = voter_params or VoterParams()

    def create_voter(self, data_params: 'DataParams') -> MultiModelVoter:
        # Cut non text processors (first two)
        post_proc = [Data.data_processor_factory().create_sequence(
            data.params().post_processors_.sample_processors[2:], data.params(), PipelineMode.Prediction) for
            data in self.datas]
        pre_proc = Data.data_processor_factory().create_sequence(
            self.data.params().pre_processors_.sample_processors, self.data.params(),
            PipelineMode.Prediction)
        out_to_in_transformer = OutputToInputTransformer(pre_proc)
        return CalamariMultiModelVoter(self.voter_params, self.datas, post_proc, out_to_in_transformer)
